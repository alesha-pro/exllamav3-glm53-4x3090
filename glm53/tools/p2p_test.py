#!/usr/bin/env python3
"""Cross-process GPU peer-to-peer mailbox test (CUDA IPC + lazy peer access), the mechanism behind a device-side
tensor-parallel all-reduce.

  p2p_test.py --dev 0 1            two processes, one per device: each exports a mailbox, opens the peer's, then
                                   they ping-pong a counter through kernels that write straight into the peer's
                                   device memory and spin on their own.
Reports: whether the peer mapping works, VRAM each process holds on each GPU, round-trip latency.
"""
import argparse, os, sys, time, struct
import multiprocessing as mp

SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstring>

static inline void ck(cudaError_t e, const char* what) { TORCH_CHECK(e == cudaSuccess, what, ": ", cudaGetErrorString(e)); }

// own mailbox: raw cudaMalloc so the pointer is an allocation base (IPC requirement)
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
    TORCH_CHECK(s.size() == sizeof(cudaIpcMemHandle_t), "bad handle size");
    cudaIpcMemHandle_t h; memcpy(&h, s.data(), sizeof(h));
    void* p = nullptr;
    ck(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess), "cudaIpcOpenMemHandle");
    return (uintptr_t) p;
}

__device__ __forceinline__ unsigned long long gtimer() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

// mode 0: write message `expect` (payload + counter) into the peer mailbox, then spin on the own mailbox until the
// peer answers with expect + 1.  mode 1: spin until message expect - 1 has arrived, then write `expect`.
// stats[0] += ns spent waiting, stats[1] += 1, stats[2] += timeouts
__global__ void pingpong_k(volatile unsigned int* mine, volatile unsigned int* peer, const float* payload, int n,
                           unsigned int expect, unsigned long long* stats, int mode)
{
    int t = threadIdx.x;
    __shared__ int ok;
    if (t == 0)
    {
        ok = 1;
        if (mode == 1)
        {
            unsigned long long t0 = gtimer(), deadline = t0 + 5000000000ull;
            while (mine[0] < expect - 1 && gtimer() < deadline) { }
            stats[0] += gtimer() - t0; stats[1] += 1;
            if (mine[0] < expect - 1) { stats[2] += 1; ok = 0; }
        }
    }
    __syncthreads();
    float* pd = (float*) (peer + 16);
    for (int i = t; i < n; i += blockDim.x) pd[i] = payload[i];
    __syncthreads();
    if (t != 0) return;
    __threadfence_system();
    peer[0] = expect;
    __threadfence_system();
    if (mode == 0)
    {
        unsigned long long t0 = gtimer(), deadline = t0 + 5000000000ull;
        while (mine[0] < expect + 1 && gtimer() < deadline) { }
        stats[0] += gtimer() - t0; stats[1] += 1;
        if (mine[0] < expect + 1) stats[2] += 1;
    }
}

void pingpong(int device, uintptr_t mine, uintptr_t peer, at::Tensor payload, int64_t expect, at::Tensor stats, int64_t first)
{
    ck(cudaSetDevice(device), "cudaSetDevice");
    pingpong_k<<<1, 256>>>((volatile unsigned int*) mine, (volatile unsigned int*) peer, payload.data_ptr<float>(),
                           (int) payload.numel(), (unsigned int) expect, (unsigned long long*) stats.data_ptr<int64_t>(), (int) first);   // first = mode
    ck(cudaGetLastError(), "launch");
}

__global__ void peek_k(const volatile unsigned int* p, unsigned int* out, const float* pd, float* outf) { out[0] = p[0]; outf[0] = pd[0]; outf[1] = pd[100]; }
py::tuple peek(int device, uintptr_t p)
{
    ck(cudaSetDevice(device), "cudaSetDevice");
    unsigned int* o; float* f;
    ck(cudaMalloc((void**) &o, 16), "m"); ck(cudaMalloc((void**) &f, 16), "m");
    peek_k<<<1, 1>>>((const volatile unsigned int*) p, o, (const float*) ((unsigned int*) p + 16), f);
    unsigned int ho; float hf[2];
    ck(cudaMemcpy(&ho, o, 4, cudaMemcpyDeviceToHost), "c"); ck(cudaMemcpy(hf, f, 8, cudaMemcpyDeviceToHost), "c");
    return py::make_tuple(ho, hf[0], hf[1]);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mbox_create", &mbox_create); m.def("mbox_open", &mbox_open); m.def("pingpong", &pingpong); m.def("peek", &peek);
}
"""


def ext():
    from torch.utils.cpp_extension import load
    d = "/tmp/p2p_test_build"
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "p2p_test.cu")
    if not os.path.exists(src) or open(src).read() != SRC:
        open(src, "w").write(SRC)
    return load(name="p2p_test_ext", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)


def worker(rank, dev, peer_dev, conn, n_hops, payload_n):
    import torch
    e = ext()
    torch.cuda.set_device(dev)
    mine, h = e.mbox_create(dev, 1 << 20)
    conn.send(h)
    hp = conn.recv()
    try:
        peer = e.mbox_open(dev, hp)
    except Exception as ex:
        conn.send(("ERR", str(ex)[:300])); return
    conn.send(("OPEN", hex(peer)))
    conn.recv()   # go
    payload = torch.full((payload_n,), float(rank + 1), device=f"cuda:{dev}")
    stats = torch.zeros(4, dtype=torch.int64, device=f"cuda:{dev}")
    # rank 0 sends 1, waits for 2, sends 3, ...; rank 1 waits for 1, sends 2, ...
    t0 = time.perf_counter()
    for i in range(n_hops):
        if rank == 0:
            e.pingpong(dev, mine, peer, payload, 2 * i + 1, stats, 0)       # write 2i+1, wait for 2i+2
        else:
            e.pingpong(dev, mine, peer, payload, 2 * i + 2, stats, 1)       # wait for 2i+1, write 2i+2
    torch.cuda.synchronize(dev)
    dt = time.perf_counter() - t0
    s = stats.tolist()
    conn.send(("DONE", rank, dt, s, e.peek(dev, mine)))
    conn.recv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, nargs=2, default=[0, 1])
    ap.add_argument("--hops", type=int, default=20000)
    ap.add_argument("--payload", type=int, default=4096 * 4)
    a = ap.parse_args()
    ext()   # build once in the parent
    mp.set_start_method("spawn")
    pa, ca = mp.Pipe(); pb, cb = mp.Pipe()
    A = mp.Process(target=worker, args=(0, a.dev[0], a.dev[1], ca, a.hops, a.payload))
    B = mp.Process(target=worker, args=(1, a.dev[1], a.dev[0], cb, a.hops, a.payload))
    A.start(); B.start()
    ha, hb = pa.recv(), pb.recv()
    pa.send(hb); pb.send(ha)
    ra, rb = pa.recv(), pb.recv()
    print("open:", ra, rb, flush=True)
    if ra[0] != "OPEN" or rb[0] != "OPEN":
        A.kill(); B.kill(); return
    os.system("nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader")
    print("pids", A.pid, B.pid, flush=True)
    # rank 1 must be queued first? both kernels just spin; start both
    pb.send("go"); pa.send("go")
    da, db = pa.recv(), pb.recv()
    for d in (da, db):
        _, rank, dt, s, pk = d
        print(f"rank {rank}: {a.hops} hops in {dt * 1e3:.1f} ms = {dt / a.hops * 1e6:.2f} us/hop host; "
              f"device wait {s[0] / max(1, s[1]) / 1e3:.2f} us/hop, timeouts {s[2]}, mailbox counter {pk[0]}, payload {pk[1]:.1f} {pk[2]:.1f}")
    pa.send("bye"); pb.send("bye")
    A.join(); B.join()


if __name__ == "__main__":
    main()
