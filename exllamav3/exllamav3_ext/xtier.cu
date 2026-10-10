#include "xtier.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

__global__ void xt_step_kernel
(
    const int64_t* __restrict__ sel, int n_picks, int first, int li, int E, int EM, int S,
    int* slotof_all, const int64_t* home_all, const int64_t* tabs, int* owner, int* refb, int* pin, int* ctl,
    int* jobs, int maxj, unsigned long long* stats, long long arena, long long rec, long long off_u, long long off_d
)
{
    __shared__ unsigned mask[32];
    __shared__ int misses[1024];
    __shared__ int nmiss, nhit, epoch;
    int t = threadIdx.x;
    if (t < 32) mask[t] = 0;
    if (t == 0) { nmiss = 0; nhit = 0; epoch = ctl[1] + 1; ctl[1] = epoch; ctl[2] = 0; }
    __syncthreads();
    for (int i = t; i < n_picks; i += blockDim.x)
    {
        long long e = sel[i] - first;
        if (e >= 0 && e < E) atomicOr(&mask[e >> 5], 1u << (e & 31));
    }
    __syncthreads();
    int* slotof = slotof_all + (long long) li * EM;
    for (int e = t; e < E; e += blockDim.x)
    {
        if (mask[e >> 5] & (1u << (e & 31)))
        {
            int s = slotof[e];
            if (s >= 0) { refb[s] = 1; pin[s] = epoch; atomicAdd(&nhit, 1); }
            else if (s == -2) atomicAdd(&nhit, 1);
            else { int k = atomicAdd(&nmiss, 1); misses[k] = e; }
        }
    }
    __syncthreads();
    if (t != 0) return;
    stats[0] += nhit; stats[1] += nmiss;
    if (S == 0) return;
    int hand = ctl[0], nj = 0;
    int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
    int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
    for (int k = 0; k < nmiss && nj < maxj; k++)
    {
        int e = misses[k], found = -1;
        for (int step = 0; step < 2 * S; step++)
        {
            int s = hand; hand = (hand + 1 == S) ? 0 : hand + 1;
            if (pin[s] == epoch) continue;              // in use by this call
            if (refb[s]) { refb[s] = 0; continue; }
            found = s; break;
        }
        if (found < 0) { stats[3]++; break; }
        int s = found, old = owner[s];
        if (old >= 0)
        {
            int ol = old / EM, oe = old % EM;
            slotof_all[(long long) ol * EM + oe] = -1;
            const int64_t* h = home_all + ((long long) ol * EM + oe) * 3;
            int64_t* og = (int64_t*) tabs[ol * 4 + 0]; int64_t* ou = (int64_t*) tabs[ol * 4 + 1];
            int64_t* od = (int64_t*) tabs[ol * 4 + 2]; int64_t* ogu = (int64_t*) tabs[ol * 4 + 3];
            og[oe] = h[0]; ou[oe] = h[1]; od[oe] = h[2];
            if (ogu) { ogu[2 * oe] = h[0]; ogu[2 * oe + 1] = h[1]; }
            stats[4]++;
        }
        owner[s] = li * EM + e; slotof[e] = s; refb[s] = 1; pin[s] = epoch;
        long long a = arena + (long long) s * rec;
        pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
        if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
        jobs[2 * nj] = e; jobs[2 * nj + 1] = s; nj++;
        stats[2]++;
    }
    ctl[0] = hand; ctl[2] = nj;
}

__global__ void xt_copy_kernel
(
    const int* __restrict__ ctl, const int* __restrict__ jobs, int li, int EM,
    const int64_t* __restrict__ home_all, long long arena, long long rec, long long vg, long long vu, long long vd,
    long long off_u, long long off_d
)
{
    long long nj = ctl[2];
    long long vpj = vg + vu + vd;
    long long total = nj * vpj;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (long long base = (long long) blockIdx.x * blockDim.x + threadIdx.x; base < total; base += 4 * stride)
    {
        int4 v[4]; int4* dst[4]; int n = 0;
        #pragma unroll
        for (int u = 0; u < 4; u++)
        {
            long long idx = base + u * stride;
            if (idx >= total) break;
            long long j = idx / vpj, r = idx - j * vpj;
            int e = jobs[2 * j], s = jobs[2 * j + 1];
            const int64_t* h = home_all + ((long long) li * EM + e) * 3;
            const int4* src; long long doff;
            if (r < vg) { src = (const int4*) h[0] + r; doff = r * 16; }
            else if (r < vg + vu) { src = (const int4*) h[1] + (r - vg); doff = off_u + (r - vg) * 16; }
            else { src = (const int4*) h[2] + (r - vg - vu); doff = off_d + (r - vg - vu) * 16; }
            v[u] = *src;
            dst[u] = (int4*) (arena + (long long) s * rec + doff);
            n++;
        }
        for (int u = 0; u < n; u++) *dst[u] = v[u];
    }
}

void xt_step
(
    at::Tensor sel, int64_t first, int64_t li, int64_t E, int64_t EM, int64_t S, at::Tensor slotof, at::Tensor home,
    at::Tensor tabs, at::Tensor owner, at::Tensor refb, at::Tensor pin, at::Tensor ctl, at::Tensor jobs,
    at::Tensor stats, int64_t arena, int64_t rec, int64_t off_u, int64_t off_d
)
{
    const at::cuda::OptionalCUDAGuard device_guard(sel.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(sel.dtype() == at::kLong && sel.is_contiguous(), "xt_step: sel must be contiguous int64");
    TORCH_CHECK(E <= 1024, "xt_step: at most 1024 local experts");
    xt_step_kernel<<<1, 256, 0, stream>>>(
        sel.data_ptr<int64_t>(), (int) sel.numel(), (int) first, (int) li, (int) E, (int) EM, (int) S,
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), owner.data_ptr<int>(),
        refb.data_ptr<int>(), pin.data_ptr<int>(), ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) (jobs.numel() / 2),
        (unsigned long long*) stats.data_ptr<int64_t>(), arena, rec, off_u, off_d);
    cuda_check(cudaPeekAtLastError());
}

void xt_copy
(
    at::Tensor ctl, at::Tensor jobs, int64_t li, int64_t EM, at::Tensor home, int64_t arena, int64_t rec,
    int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d, int64_t grid
)
{
    const at::cuda::OptionalCUDAGuard device_guard(ctl.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    xt_copy_kernel<<<(int) grid, 256, 0, stream>>>(
        ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) li, (int) EM, home.data_ptr<int64_t>(), arena, rec,
        sg / 16, su / 16, sd / 16, off_u, off_d);
    cuda_check(cudaPeekAtLastError());
}
