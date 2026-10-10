#include <cuda_fp16.h>
#include "p2p.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstring>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"

#define P2P_THREADS 256
#define P2P_FLAG_OFF 64
#define P2P_FLAG_STRIDE 64

#define P2P_DT_FLOAT 0
#define P2P_DT_HALF 1
#define P2P_DT_BF16 2
#define P2P_DT_RAW 3

static inline size_t p2p_data_off_host(int ndev)
{
    size_t o = P2P_FLAG_OFF + (size_t) ndev * P2P_NSLOT * P2P_FLAG_STRIDE;
    return (o + 4095) / 4096 * 4096;
}

size_t pg_p2p_mailbox_size(int ndev)
{
    return p2p_data_off_host(ndev) + (size_t) ndev * P2P_NSLOT * P2P_SLOT_BYTES;
}

uintptr_t pg_p2p_create(int device, int ndev, std::string& handle_out)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    void* p = nullptr;
    size_t nbytes = pg_p2p_mailbox_size(ndev);
    cuda_check(cudaMalloc(&p, nbytes));
    cuda_check(cudaMemset(p, 0, nbytes));
    cudaIpcMemHandle_t h;
    cuda_check(cudaIpcGetMemHandle(&h, p));
    handle_out.assign((const char*) &h, sizeof(h));
    return (uintptr_t) p;
}

uintptr_t pg_p2p_open(int device, const std::string& handle)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    TORCH_CHECK(handle.size() == sizeof(cudaIpcMemHandle_t), "pg_p2p_open: bad handle size");
    cudaIpcMemHandle_t h;
    memcpy(&h, handle.data(), sizeof(h));
    void* p = nullptr;
    cudaError_t e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(e == cudaSuccess, "pg_p2p_open: cudaIpcOpenMemHandle failed (no peer access between the "
                "devices?): ", cudaGetErrorString(e));
    return (uintptr_t) p;
}

bool pg_p2p_fits(const at::Tensor& tensor)
{
    size_t nbytes = tensor.numel() * tensor.element_size();
    return nbytes > 0 && nbytes <= P2P_SLOT_BYTES && nbytes % 16 == 0 && tensor.is_contiguous();
}

struct P2PArgs
{
    uint8_t* mine;
    uint8_t* peer[MAX_DEVICES];
    PGContext* ctx;
    uint8_t* data;
    uint32_t* abort_flag;
    uint32_t nbytes;
    uint32_t mask;
    int me;
    int ndev;
    int contributor;
};

__device__ __forceinline__ float p2p_bf16_to_float(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }
__device__ __forceinline__ uint16_t p2p_float_to_bf16(float f) { return (uint16_t) ((__float_as_uint(f) + 0x8000u) >> 16); }

template <int dtype>
__global__ __launch_bounds__(P2P_THREADS)
void pg_all_reduce_p2p_kernel(P2PArgs a)
{
    const int t = threadIdx.x;
    __shared__ uint32_t s_stage;
    __shared__ uint32_t s_contrib;
    __shared__ uint32_t s_abort;

    if (t == 0)
    {
        s_stage = *(const volatile uint32_t*) a.mine;
        s_abort = 0;
    }
    __syncthreads();

    const uint32_t stage = s_stage;
    const uint32_t slot = stage % P2P_NSLOT;
    const uint32_t next = (stage + 1u) & 0x7fffffffu;
    size_t data_off = P2P_FLAG_OFF + (size_t) a.ndev * P2P_NSLOT * P2P_FLAG_STRIDE;
    data_off = (data_off + 4095) / 4096 * 4096;
    const uint32_t n16 = a.nbytes / 16;
    uint4* own = (uint4*) a.data;

    // Send: payload into every peer's mailbox, then the stage flag behind it
    if (a.contributor)
    {
        for (int q = 0; q < a.ndev; ++q)
        {
            if (q == a.me || !((a.mask >> q) & 1u)) continue;
            uint4* dst = (uint4*) (a.peer[q] + data_off + (size_t) (a.me * P2P_NSLOT + slot) * P2P_SLOT_BYTES);
            for (uint32_t i = t; i < n16; i += P2P_THREADS)
                dst[i] = own[i];
        }
    }
    __syncthreads();

    if (t == 0)
    {
        __threadfence_system();
        const uint32_t v = next | (a.contributor ? 0u : 0x80000000u);
        for (int q = 0; q < a.ndev; ++q)
        {
            if (q == a.me || !((a.mask >> q) & 1u)) continue;
            stg_release_sys_u32((uint32_t*) (a.peer[q] + P2P_FLAG_OFF + (size_t) (a.me * P2P_NSLOT + slot) * P2P_FLAG_STRIDE), v);
        }
        *(volatile uint32_t*) a.mine = next;

        // Receive: wait for every peer's flag of this stage in the own mailbox
        uint32_t contrib = a.contributor ? (1u << a.me) : 0u;
        const uint64_t deadline = sync_deadline();
        for (int q = 0; q < a.ndev && !s_abort; ++q)
        {
            if (q == a.me || !((a.mask >> q) & 1u)) continue;
            const uint32_t* f = (const uint32_t*) (a.mine + P2P_FLAG_OFF + (size_t) (q * P2P_NSLOT + slot) * P2P_FLAG_STRIDE);
            uint64_t sleep = SYNC_MIN_SLEEP;
            while (true)
            {
                uint32_t fv = ldg_acquire_sys_u32(f);
                if ((fv & 0x7fffffffu) == next)
                {
                    if (!(fv >> 31)) contrib |= 1u << q;
                    break;
                }
                __nanosleep(sleep);
                if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
                else if (check_timeout(a.ctx, deadline, "pg_all_reduce_p2p_kernel"))
                {
                    *a.abort_flag = 1;
                    s_abort = 1;
                    break;
                }
            }
        }
        s_contrib = contrib;
    }
    __syncthreads();
    if (s_abort) return;

    const uint32_t contrib = s_contrib;
    if (contrib == (1u << a.me) || contrib == 0u) return;      // nothing arrives: own data stands

    const uint8_t* base = a.mine + data_off + (size_t) slot * P2P_SLOT_BYTES;
    #define P2P_SRC(q) ((const uint4*) (base + (size_t) (q) * P2P_NSLOT * P2P_SLOT_BYTES))

    // Single contributor: verbatim copy (broadcast), any dtype
    if (__popc(contrib) == 1)
    {
        const int q = __ffs(contrib) - 1;
        const uint4* src = P2P_SRC(q);
        for (uint32_t i = t; i < n16; i += P2P_THREADS)
            own[i] = src[i];
        return;
    }

    // Sum in rank order
    for (uint32_t i = t; i < n16; i += P2P_THREADS)
    {
        if constexpr (dtype == P2P_DT_FLOAT)
        {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            bool first = true;
            for (int q = 0; q < a.ndev; ++q)
            {
                if (!((contrib >> q) & 1u)) continue;
                const float4 v = (q == a.me) ? *(const float4*) (own + i) : *(const float4*) (P2P_SRC(q) + i);
                if (first) { acc = v; first = false; }
                else { acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
            }
            *(float4*) (own + i) = acc;
        }
        else if constexpr (dtype == P2P_DT_HALF || dtype == P2P_DT_BF16)
        {
            float acc[8];
            bool first = true;
            for (int q = 0; q < a.ndev; ++q)
            {
                if (!((contrib >> q) & 1u)) continue;
                const uint4 v = (q == a.me) ? own[i] : P2P_SRC(q)[i];
                const uint16_t* h = (const uint16_t*) &v;
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    float f;
                    if constexpr (dtype == P2P_DT_HALF) f = __half2float(__ushort_as_half(h[k]));
                    else f = p2p_bf16_to_float(h[k]);
                    acc[k] = first ? f : acc[k] + f;
                }
                first = false;
            }
            uint4 o;
            uint16_t* ho = (uint16_t*) &o;
            #pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                if constexpr (dtype == P2P_DT_HALF) ho[k] = __half_as_ushort(__float2half_rn(acc[k]));
                else ho[k] = p2p_float_to_bf16(acc[k]);
            }
            own[i] = o;
        }
        else
        {
            // Raw payloads cannot be summed: flag it (every rank sees the same contributor set)
            *a.abort_flag = 1;
        }
    }
    #undef P2P_SRC
}

void pg_all_reduce_p2p
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t mailbox,
    std::vector<uintptr_t> peers,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);
    TORCH_CHECK(pg_p2p_fits(tensor), "pg_all_reduce_p2p: payload does not fit the p2p path");

    P2PArgs a;
    a.mine = (uint8_t*) mailbox;
    a.mask = 0;
    a.ndev = 0;
    for (int i = 0; i < MAX_DEVICES; ++i)
        a.peer[i] = (i < (int) peers.size()) ? (uint8_t*) peers[i] : nullptr;
    for (uintptr_t d : devices)
    {
        TORCH_CHECK(d < MAX_DEVICES, "pg_all_reduce_p2p: device index out of range");
        a.mask |= 1u << d;
        if ((int) d + 1 > a.ndev) a.ndev = (int) d + 1;
        TORCH_CHECK((int) d == this_device || a.peer[d] != nullptr, "pg_all_reduce_p2p: peer mailbox not mapped");
    }
    a.ctx = (PGContext*) ctx_dev;
    a.data = (uint8_t*) tensor.data_ptr();
    a.abort_flag = (uint32_t*) abort_flag.data_ptr();
    a.nbytes = (uint32_t) (tensor.numel() * tensor.element_size());
    a.me = this_device;
    a.contributor = contributor ? 1 : 0;

    if (tensor.dtype() == at::kFloat)
        pg_all_reduce_p2p_kernel<P2P_DT_FLOAT><<<1, P2P_THREADS, 0, stream>>>(a);
    else if (tensor.dtype() == at::kHalf)
        pg_all_reduce_p2p_kernel<P2P_DT_HALF><<<1, P2P_THREADS, 0, stream>>>(a);
    else if (tensor.dtype() == at::kBFloat16)
        pg_all_reduce_p2p_kernel<P2P_DT_BF16><<<1, P2P_THREADS, 0, stream>>>(a);
    else
        pg_all_reduce_p2p_kernel<P2P_DT_RAW><<<1, P2P_THREADS, 0, stream>>>(a);
    cuda_check(cudaPeekAtLastError());
}
