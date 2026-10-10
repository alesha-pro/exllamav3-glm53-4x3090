#!/usr/bin/env python3
"""Bulk GPU-to-GPU transfer between processes (CUDA IPC + lazy peer access): how fast can one rank of a
tensor-parallel model push a large tensor into another rank's device memory, the transport a device-side
all-reduce of prefill-sized tensors would use.

  p2p_bulk.py [--mb 32 64] [--reps 20] [--dev 0 1 2 3]

Per payload size it reports
  pair      one sender, one receiver (0 -> 1): cudaMemcpy to the peer mapping, and a copy kernel writing into it
  ring      every rank sends to the next one at the same time
  all       every rank sends to all three peers at the same time (what a one-round all-reduce does)
  host      the same payload device -> pinned host -> device, for comparison (the native path's two hops)
"""
import argparse, os, time
import multiprocessing as mp

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>

static inline void ck(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }
static inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

py::tuple mbox_create(int device, size_t nbytes)
{
    ck(cudaSetDevice(device), "cudaSetDevice");
    void* p = nullptr;
    ck(cudaMalloc(&p, nbytes), "cudaMalloc");
    ck(cudaMemset(p, 0, nbytes), "cudaMemset");
    cudaIpcMemHandle_t h;
    ck(cudaIpcGetMemHandle(&h, p), "cudaIpcGetMemHandle");
    return py::make_tuple((uintptr_t) p, py::bytes((const char*) &h, sizeof(h)));
}

uintptr_t mbox_open(int device, py::bytes handle)
{
    ck(cudaSetDevice(device), "cudaSetDevice");
    std::string s = handle;
    cudaIpcMemHandle_t h; memcpy(&h, s.data(), sizeof(h));
    void* p = nullptr;
    ck(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
    return (uintptr_t) p;
}

static cudaStream_t streams[8]; static bool have_streams = false;
static void mk_streams() { if (!have_streams) { for (int i = 0; i < 8; i++) ck(cudaStreamCreate(&streams[i]), "stream"); have_streams = true; } }

// cudaMemcpy of nbytes from a local tensor to each destination pointer, one stream per destination
double send_memcpy(int device, std::vector<uintptr_t> dsts, at::Tensor src, size_t nbytes, int reps)
{
    ck(cudaSetDevice(device), "cudaSetDevice"); mk_streams();
    ck(cudaDeviceSynchronize(), "sync");
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (size_t d = 0; d < dsts.size(); d++)
            ck(cudaMemcpyAsync((void*) dsts[d], src.data_ptr(), nbytes, cudaMemcpyDeviceToDevice, streams[d]), "cudaMemcpyAsync");
    ck(cudaDeviceSynchronize(), "sync");
    return now() - t0;
}

__global__ void copy_k(uint4* dst, const uint4* src, size_t n)
{
    size_t stride = (size_t) gridDim.x * blockDim.x;
    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

// the same payload written by a kernel (posted writes into the peer mapping)
double send_kernel(int device, std::vector<uintptr_t> dsts, at::Tensor src, size_t nbytes, int reps, int blocks)
{
    ck(cudaSetDevice(device), "cudaSetDevice"); mk_streams();
    ck(cudaDeviceSynchronize(), "sync");
    double t0 = now();
    for (int r = 0; r < reps; r++)
        for (size_t d = 0; d < dsts.size(); d++)
            copy_k<<<blocks, 256, 0, streams[d]>>>((uint4*) dsts[d], (const uint4*) src.data_ptr(), nbytes / 16);
    ck(cudaGetLastError(), "launch");
    ck(cudaDeviceSynchronize(), "sync");
    return now() - t0;
}

// device -> pinned host -> device round trip of the same payload
double host_roundtrip(int device, at::Tensor src, at::Tensor dst, size_t nbytes, int reps)
{
    ck(cudaSetDevice(device), "cudaSetDevice");
    void* h = nullptr;
    ck(cudaMallocHost(&h, nbytes), "cudaMallocHost");
    ck(cudaDeviceSynchronize(), "sync");
    double t0 = now();
    for (int r = 0; r < reps; r++)
    {
        ck(cudaMemcpy(h, src.data_ptr(), nbytes, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(dst.data_ptr(), h, nbytes, cudaMemcpyHostToDevice), "h2d");
    }
    double dt = now() - t0;
    cudaFreeHost(h);
    return dt;
}

uint64_t checksum(int device, uintptr_t p, size_t nbytes)
{
    ck(cudaSetDevice(device), "cudaSetDevice");
    std::vector<uint32_t> v(4);
    ck(cudaMemcpy(v.data(), (void*) (p + nbytes - 16), 16, cudaMemcpyDeviceToHost), "peek");
    return ((uint64_t) v[2] << 32) | v[3];
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mbox_create", &mbox_create); m.def("mbox_open", &mbox_open); m.def("send_memcpy", &send_memcpy);
    m.def("send_kernel", &send_kernel); m.def("host_roundtrip", &host_roundtrip); m.def("checksum", &checksum);
}
"""


def ext():
    from torch.utils.cpp_extension import load
    d = "/tmp/p2p_bulk_build"
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "p2p_bulk.cu")
    if not os.path.exists(src) or open(src).read() != SRC:
        open(src, "w").write(SRC)
    return load(name="p2p_bulk_ext", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)


def worker(rank, dev, ndev, conn, max_bytes):
    import torch
    e = ext()
    torch.cuda.set_device(dev)
    mine, h = e.mbox_create(dev, max_bytes * ndev)          # one slot per source rank
    conn.send(h)
    handles = conn.recv()
    peers = {}
    try:
        for r, hp in handles.items():
            if r != rank:
                peers[r] = e.mbox_open(dev, hp)
    except Exception as ex:
        conn.send(("ERR", str(ex)[:300])); return
    conn.send(("OPEN", len(peers)))
    src = torch.arange(max_bytes // 4, dtype=torch.int32, device=f"cuda:{dev}") + rank * 7
    dst_local = torch.empty_like(src)
    while True:
        cmd = conn.recv()
        if cmd[0] == "bye":
            return
        op, targets, nbytes, reps, start_at, blocks = cmd
        dsts = [peers[t] + rank * max_bytes for t in targets]
        while time.time() < start_at:
            pass
        t0 = time.time()
        if op == "memcpy":
            dt = e.send_memcpy(dev, dsts, src, nbytes, reps)
        elif op == "kernel":
            dt = e.send_kernel(dev, dsts, src, nbytes, reps, blocks)
        elif op == "host":
            dt = e.host_roundtrip(dev, src, dst_local, nbytes, reps)
        else:
            dt = 0.0
        conn.send((rank, dt, t0, time.time()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--mb", type=float, nargs="+", default=[32, 64])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, nargs="+", default=[64, 512])
    a = ap.parse_args()
    ext()
    mp.set_start_method("spawn")
    n = len(a.dev)
    max_bytes = int(max(a.mb) * 2**20) // 16 * 16
    pipes, procs = [], []
    for r, d in enumerate(a.dev):
        pa, ca = mp.Pipe()
        p = mp.Process(target=worker, args=(r, d, n, ca, max_bytes)); p.start()
        pipes.append(pa); procs.append(p)
    handles = {r: pipes[r].recv() for r in range(n)}
    for r in range(n):
        pipes[r].send(handles)
    st = [pipes[r].recv() for r in range(n)]
    print("open:", st, flush=True)
    if any(s[0] != "OPEN" for s in st):
        for p in procs: p.kill()
        return

    def run(name, op, plan, nbytes, blocks=0):
        """plan: {sender rank: [target ranks]}"""
        start = time.time() + 0.3
        for r, targets in plan.items():
            pipes[r].send((op, targets, nbytes, a.reps, start, blocks))
        res = [pipes[r].recv() for r in plan]
        wall = max(x[3] for x in res) - start
        per = []
        for (r, dt, t0, t1), targets in zip(res, plan.values()):
            k = max(1, len(targets)) if op != "host" else 2
            per.append(nbytes * a.reps * k / dt / 1e9)
        sends = sum(len(t) if op != "host" else 1 for t in plan.values())
        one = wall / a.reps * 1e3
        print(f"  {name:34s} per-rank out {min(per):5.2f}-{max(per):5.2f} GB/s, one round (all transfers of a collective) {one:6.2f} ms", flush=True)

    for mb in a.mb:
        nbytes = int(mb * 2**20) // 16 * 16
        print(f"== payload {mb} MiB, {a.reps} reps", flush=True)
        run("pair 0->1 memcpy", "memcpy", {0: [1]}, nbytes)
        for b in a.blocks:
            run(f"pair 0->1 kernel {b} blocks", "kernel", {0: [1]}, nbytes, b)
        if n >= 4:
            run("pairs 0<->1, 2<->3 memcpy", "memcpy", {0: [1], 1: [0], 2: [3], 3: [2]}, nbytes)
            run("pairs 0<->2, 1<->3 memcpy", "memcpy", {0: [2], 2: [0], 1: [3], 3: [1]}, nbytes)
            run("ring r->r+1 memcpy", "memcpy", {r: [(r + 1) % n] for r in range(n)}, nbytes)
            run("all -> 3 peers memcpy", "memcpy", {r: [t for t in range(n) if t != r] for r in range(n)}, nbytes)
            run(f"all -> 3 peers kernel {a.blocks[-1]} blocks", "kernel", {r: [t for t in range(n) if t != r] for r in range(n)}, nbytes, a.blocks[-1])
            run("3 -> rank 0 memcpy (gather)", "memcpy", {r: [0] for r in range(1, n)}, nbytes)
            run("rank 0 -> 3 peers memcpy (bcast)", "memcpy", {0: [1, 2, 3]}, nbytes)
        run("host round trip, 1 rank", "host", {0: []}, nbytes)
        run("host round trip, all ranks", "host", {r: [] for r in range(n)}, nbytes)
    for r in range(n):
        pipes[r].send(("bye",))
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
