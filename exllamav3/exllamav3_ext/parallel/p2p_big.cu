#include <cuda_fp16.h>
#include "p2p_big.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "context.cuh"
#include "timeout.cuh"

#define P2PB_THREADS 256
#define P2PB_MAX_BLOCKS 512

#define P2PB_DT_FLOAT 0
#define P2PB_DT_HALF 1
#define P2PB_DT_BF16 2

size_t pg_p2pb_mailbox_size(size_t slot_bytes, int rounds, size_t bcast_bytes)
{
    return P2PB_HEADER + (size_t) 3 * rounds * slot_bytes + bcast_bytes;
}

uintptr_t pg_p2pb_create(int device, size_t slot_bytes, int rounds, size_t bcast_bytes, std::string& handle_out)
{
    const at::cuda::OptionalCUDAGuard device_guard(device);
    TORCH_CHECK(rounds >= 1 && rounds <= P2PB_MAX_ROUNDS, "pg_p2pb_create: 2, 4 or 8 ranks only");
    TORCH_CHECK(slot_bytes % 4096 == 0, "pg_p2pb_create: slot size must be a multiple of 4096");
    void* p = nullptr;
    TORCH_CHECK(bcast_bytes % 4096 == 0, "pg_p2pb_create: broadcast region must be a multiple of 4096");
    size_t nbytes = pg_p2pb_mailbox_size(slot_bytes, rounds, bcast_bytes);
    cuda_check(cudaMalloc(&p, nbytes));
    cuda_check(cudaMemset(p, 0, P2PB_HEADER));
    cudaIpcMemHandle_t h;
    cuda_check(cudaIpcGetMemHandle(&h, p));
    handle_out.assign((const char*) &h, sizeof(h));
    return (uintptr_t) p;
}

bool pg_p2pb_fits(const at::Tensor& tensor, size_t slot_bytes, size_t min_bytes)
{
    if (!tensor.is_contiguous()) return false;
    if (tensor.dtype() != at::kFloat && tensor.dtype() != at::kHalf && tensor.dtype() != at::kBFloat16) return false;
    size_t n = tensor.numel();
    size_t wire = n * 2;
    return n > 0 && n % 8 == 0 && wire >= min_bytes && wire <= slot_bytes;
}

__device__ __forceinline__ float p2pb_bf16_to_float(uint16_t b) { return __uint_as_float(((uint32_t) b) << 16); }
__device__ __forceinline__ uint16_t p2pb_float_to_bf16(float f) { return (uint16_t) ((__float_as_uint(f) + 0x8000u) >> 16); }

template <int dtype>
__device__ __forceinline__ float p2pb_from_wire(uint16_t h)
{
    if constexpr (dtype == P2PB_DT_HALF) return __half2float(__ushort_as_half(h));
    else return p2pb_bf16_to_float(h);
}

template <int dtype>
__device__ __forceinline__ uint16_t p2pb_to_wire(float f)
{
    if constexpr (dtype == P2PB_DT_HALF) return __half_as_ushort(__float2half_rn(f));
    else return p2pb_float_to_bf16(f);
}

// Own contribution -> wire buffer (zeros for a rank that does not contribute). n8 counts units of 8 elements
template <int dtype>
__global__ __launch_bounds__(P2PB_THREADS)
void p2pb_pack_kernel(const uint8_t* __restrict__ x, uint4* __restrict__ w, uint32_t n8, int contributor)
{
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n8; i += stride)
    {
        uint4 o = make_uint4(0u, 0u, 0u, 0u);
        if (contributor)
        {
            if constexpr (dtype == P2PB_DT_FLOAT)
            {
                // Round fp32 -> bf16 to nearest, as the CPU path's wire does
                const uint4 a = ((const uint4*) x)[2 * (size_t) i];
                const uint4 b = ((const uint4*) x)[2 * (size_t) i + 1];
                o.x = ((a.y + 0x8000u) & 0xffff0000u) | ((a.x + 0x8000u) >> 16);
                o.y = ((a.w + 0x8000u) & 0xffff0000u) | ((a.z + 0x8000u) >> 16);
                o.z = ((b.y + 0x8000u) & 0xffff0000u) | ((b.x + 0x8000u) >> 16);
                o.w = ((b.w + 0x8000u) & 0xffff0000u) | ((b.z + 0x8000u) >> 16);
            }
            else
            {
                o = ((const uint4*) x)[i];
            }
        }
        w[i] = o;
    }
}

struct P2PBSum
{
    const uint4* w;             // own running sum (wire)
    const uint4* inbox;         // the partner's running sum (wire), in the own mailbox
    uint8_t* out;               // next wire buffer, or the tensor itself in the last round
    uint32_t* remote_flag;      // this round's flag in the partner's mailbox
    const uint32_t* my_flag;    // this round's flag in the own mailbox
    PGContext* ctx;
    uint32_t* abort_flag;
    uint32_t n8;
    uint32_t flag_value;
};

// One round: tell the partner the payload has landed (the copy that carried it is earlier on this stream),
// wait for the partner's payload, add. final_ = 1 writes the tensor in its own dtype
template <int dtype, int final_>
__global__ __launch_bounds__(P2PB_THREADS)
void p2pb_sum_kernel(P2PBSum a)
{
    const int t = threadIdx.x;
    __shared__ uint32_t s_abort;

    if (t == 0)
    {
        s_abort = 0;
        if (blockIdx.x == 0)
        {
            __threadfence_system();
            stg_release_sys_u32(a.remote_flag, a.flag_value);
        }
        const uint64_t deadline = sync_deadline();
        uint64_t sleep = SYNC_MIN_SLEEP;
        while (ldg_acquire_sys_u32(a.my_flag) != a.flag_value)
        {
            __nanosleep(sleep);
            if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
            else if (check_timeout(a.ctx, deadline, "pg_all_reduce_p2p_big"))
            {
                *a.abort_flag = 1;
                s_abort = 1;
                break;
            }
        }
    }
    __syncthreads();
    if (s_abort) return;

    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + t; i < a.n8; i += stride)
    {
        const uint4 u = a.w[i];
        const uint4 v = a.inbox[i];
        const uint16_t* hu = (const uint16_t*) &u;
        const uint16_t* hv = (const uint16_t*) &v;
        float f[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            f[k] = p2pb_from_wire<dtype>(hu[k]) + p2pb_from_wire<dtype>(hv[k]);

        if constexpr (final_ && dtype == P2PB_DT_FLOAT)
        {
            float4* o = (float4*) a.out + 2 * (size_t) i;
            o[0] = make_float4(f[0], f[1], f[2], f[3]);
            o[1] = make_float4(f[4], f[5], f[6], f[7]);
        }
        else
        {
            uint4 o;
            uint16_t* ho = (uint16_t*) &o;
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                ho[k] = p2pb_to_wire<dtype>(f[k]);
            ((uint4*) a.out)[i] = o;
        }
    }
}

static std::mutex g_p2pb_mutex;
static std::unordered_map<uintptr_t, uint32_t> g_p2pb_stage;

template <int dtype>
static void p2pb_launch_sum(bool final_, const P2PBSum& a, int blocks, cudaStream_t stream)
{
    if (final_) p2pb_sum_kernel<dtype, 1><<<blocks, P2PB_THREADS, 0, stream>>>(a);
    else        p2pb_sum_kernel<dtype, 0><<<blocks, P2PB_THREADS, 0, stream>>>(a);
}

void pg_all_reduce_p2p_big
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t mailbox,
    std::vector<uintptr_t> peers,
    size_t slot_bytes,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    // Rank index = position in the sorted device list (the same on every rank)
    std::vector<int> devs(devices.begin(), devices.end());
    std::sort(devs.begin(), devs.end());
    const int nranks = (int) devs.size();
    int rounds = 0;
    while ((1 << rounds) < nranks) ++rounds;
    TORCH_CHECK((1 << rounds) == nranks && rounds >= 1 && rounds <= P2PB_MAX_ROUNDS,
                "pg_all_reduce_p2p_big: needs 2, 4 or 8 ranks");
    int idx = -1;
    for (int i = 0; i < nranks; ++i) if (devs[i] == this_device) idx = i;
    TORCH_CHECK(idx >= 0, "pg_all_reduce_p2p_big: this device is not in the group");

    const size_t n = tensor.numel();
    const size_t wire_bytes = n * 2;
    TORCH_CHECK(n % 8 == 0 && wire_bytes <= slot_bytes, "pg_all_reduce_p2p_big: payload does not fit the p2p path");
    const uint32_t n8 = (uint32_t) (n / 8);
    const int blocks = (int) std::min<size_t>(P2PB_MAX_BLOCKS, std::max<size_t>(1, (n8 + P2PB_THREADS * 8 - 1) / (P2PB_THREADS * 8)));

    uint32_t stage;
    {
        std::lock_guard<std::mutex> lock(g_p2pb_mutex);
        stage = g_p2pb_stage[mailbox]++;
    }
    const uint32_t parity = stage & 1u;

    uint8_t* mine = (uint8_t*) mailbox;
    uint8_t* data = (uint8_t*) tensor.data_ptr();
    auto slot_of = [&](uint8_t* base, int s) { return base + P2PB_HEADER + (size_t) s * slot_bytes; };

    int dt;
    if (tensor.dtype() == at::kFloat) dt = P2PB_DT_FLOAT;
    else if (tensor.dtype() == at::kHalf) dt = P2PB_DT_HALF;
    else if (tensor.dtype() == at::kBFloat16) dt = P2PB_DT_BF16;
    else TORCH_CHECK(false, "pg_all_reduce_p2p_big: unsupported dtype");

    // Own contribution -> wire
    uint4* w0 = (uint4*) slot_of(mine, 0);
    if (dt == P2PB_DT_FLOAT) p2pb_pack_kernel<P2PB_DT_FLOAT><<<blocks, P2PB_THREADS, 0, stream>>>(data, w0, n8, contributor ? 1 : 0);
    else                     p2pb_pack_kernel<P2PB_DT_BF16><<<blocks, P2PB_THREADS, 0, stream>>>(data, w0, n8, contributor ? 1 : 0);

    for (int j = 0; j < rounds; ++j)
    {
        const int partner = devs[idx ^ (1 << j)];
        TORCH_CHECK(partner < (int) peers.size() && peers[partner] != 0, "pg_all_reduce_p2p_big: partner mailbox not mapped");
        uint8_t* theirs = (uint8_t*) peers[partner];
        const int inbox_slot = rounds + 2 * j + (int) parity;

        // Running sum -> the partner's inbox, device to device
        cuda_check(cudaMemcpyAsync(slot_of(theirs, inbox_slot), slot_of(mine, j), wire_bytes, cudaMemcpyDeviceToDevice, stream));

        const bool final_ = (j == rounds - 1);
        P2PBSum a;
        a.w = (const uint4*) slot_of(mine, j);
        a.inbox = (const uint4*) slot_of(mine, inbox_slot);
        a.out = final_ ? data : slot_of(mine, j + 1);
        a.remote_flag = (uint32_t*) (theirs + (size_t) 64 * j);
        a.my_flag = (const uint32_t*) (mine + (size_t) 64 * j);
        a.ctx = (PGContext*) ctx_dev;
        a.abort_flag = (uint32_t*) abort_flag.data_ptr();
        a.n8 = n8;
        a.flag_value = stage + 1u;

        if (dt == P2PB_DT_FLOAT) p2pb_launch_sum<P2PB_DT_FLOAT>(final_, a, blocks, stream);
        else if (dt == P2PB_DT_HALF) p2pb_launch_sum<P2PB_DT_HALF>(final_, a, blocks, stream);
        else p2pb_launch_sum<P2PB_DT_BF16>(final_, a, blocks, stream);
    }
    cuda_check(cudaPeekAtLastError());
}


// ------------------------------------------------------------------------------------------------------
// Broadcast

static std::unordered_map<uintptr_t, uint32_t> g_p2pb_bstage;

bool pg_p2pb_bcast_fits(const at::Tensor& tensor, size_t bcast_bytes, size_t min_bytes)
{
    size_t nbytes = tensor.numel() * tensor.element_size();
    return tensor.is_contiguous() && nbytes > 0 && nbytes % 16 == 0 && nbytes >= min_bytes && nbytes <= bcast_bytes;
}

__global__ void p2pb_flag_kernel(uint32_t* remote_flag, uint32_t value)
{
    __threadfence_system();
    stg_release_sys_u32(remote_flag, value);
}

struct P2PBRecv
{
    const uint4* src;           // the landed payload, in the own mailbox
    uint4* dst;                 // the tensor
    const uint32_t* my_flag;
    PGContext* ctx;
    uint32_t* abort_flag;
    uint32_t n16;
    uint32_t flag_value;
};

__global__ __launch_bounds__(P2PB_THREADS)
void p2pb_recv_kernel(P2PBRecv a)
{
    const int t = threadIdx.x;
    __shared__ uint32_t s_abort;
    if (t == 0)
    {
        s_abort = 0;
        const uint64_t deadline = sync_deadline();
        uint64_t sleep = SYNC_MIN_SLEEP;
        while (ldg_acquire_sys_u32(a.my_flag) != a.flag_value)
        {
            __nanosleep(sleep);
            if (sleep < SYNC_MAX_SLEEP) sleep <<= 1;
            else if (check_timeout(a.ctx, deadline, "pg_broadcast_p2p_big"))
            {
                *a.abort_flag = 1;
                s_abort = 1;
                break;
            }
        }
    }
    __syncthreads();
    if (s_abort) return;

    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + t; i < a.n16; i += stride)
        a.dst[i] = a.src[i];
}

void pg_broadcast_p2p_big
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t mailbox,
    std::vector<uintptr_t> peers,
    size_t slot_bytes,
    size_t bcast_bytes,
    at::Tensor& abort_flag
)
{
    const at::cuda::OptionalCUDAGuard device_guard(this_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    pg_check_timeout(ctx);

    std::vector<int> devs(devices.begin(), devices.end());
    std::sort(devs.begin(), devs.end());
    const int nranks = (int) devs.size();
    int rounds = 0;
    while ((1 << rounds) < nranks) ++rounds;
    TORCH_CHECK((1 << rounds) == nranks && rounds >= 1 && rounds <= P2PB_MAX_ROUNDS,
                "pg_broadcast_p2p_big: needs 2, 4 or 8 ranks");
    int idx = -1, src_idx = -1;
    for (int i = 0; i < nranks; ++i)
    {
        if (devs[i] == this_device) idx = i;
        if (devs[i] == src_device) src_idx = i;
    }
    TORCH_CHECK(idx >= 0 && src_idx >= 0, "pg_broadcast_p2p_big: device is not in the group");

    const size_t nbytes = tensor.numel() * tensor.element_size();
    TORCH_CHECK(nbytes % 16 == 0 && nbytes <= bcast_bytes, "pg_broadcast_p2p_big: payload does not fit the p2p path");
    const uint32_t n16 = (uint32_t) (nbytes / 16);
    const int blocks = (int) std::min<size_t>(P2PB_MAX_BLOCKS, std::max<size_t>(1, (n16 + P2PB_THREADS * 8 - 1) / (P2PB_THREADS * 8)));

    uint32_t stage;
    {
        std::lock_guard<std::mutex> lock(g_p2pb_mutex);
        stage = g_p2pb_bstage[mailbox]++;
    }
    const size_t area_off = P2PB_HEADER + (size_t) 3 * rounds * slot_bytes;

    uint8_t* mine = (uint8_t*) mailbox;
    uint8_t* data = (uint8_t*) tensor.data_ptr();
    const int rel = idx ^ src_idx;      // 0 for the owner

    for (int j = 0; j < rounds; ++j)
    {
        const int partner = devs[idx ^ (1 << j)];
        if (rel < (1 << j))
        {
            // Has the data: send it to this round's partner, then raise the partner's flag behind it
            TORCH_CHECK(partner < (int) peers.size() && peers[partner] != 0, "pg_broadcast_p2p_big: partner mailbox not mapped");
            uint8_t* theirs = (uint8_t*) peers[partner];
            cuda_check(cudaMemcpyAsync(theirs + area_off, data, nbytes, cudaMemcpyDeviceToDevice, stream));
            p2pb_flag_kernel<<<1, 1, 0, stream>>>((uint32_t*) (theirs + P2PB_BFLAG_OFF + (size_t) 64 * j), stage + 1u);
        }
        else if (rel < (1 << (j + 1)))
        {
            // Receives in this round
            P2PBRecv a;
            a.src = (const uint4*) (mine + area_off);
            a.dst = (uint4*) data;
            a.my_flag = (const uint32_t*) (mine + P2PB_BFLAG_OFF + (size_t) 64 * j);
            a.ctx = (PGContext*) ctx_dev;
            a.abort_flag = (uint32_t*) abort_flag.data_ptr();
            a.n16 = n16;
            a.flag_value = stage + 1u;
            p2pb_recv_kernel<<<blocks, P2PB_THREADS, 0, stream>>>(a);
        }
    }
    cuda_check(cudaPeekAtLastError());
}
