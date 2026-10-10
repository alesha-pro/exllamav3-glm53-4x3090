#pragma once

#include <ATen/Tensor.h>

// Device-side expert cache for tiered MoE layers (util/xtier.py). Cold experts live in pinned host
// memory; a slot arena in VRAM holds the ones in use. Per layer call, after routing and before the
// expert kernels:
//   xt_step  one block, no host sync: dedup the picks; resident picks (static or cached) count as hits
//            and pin their slot; each miss takes a CLOCK victim (never a slot pinned by this call), the
//            victim's pointer-table rows go back to its host address, the miss's rows point at the slot
//            and a copy job is queued
//   xt_copy  grid-stride gather of the queued experts, host -> slot
// The expert kernels then read every expert through the live pointer tables, so the arithmetic and the
// bytes are those of an all-VRAM run; only the address differs.
// After the CLOCK cache of sybil-solutions/glm53-flash-offload (MIT), reworked for statically resident
// experts (slotof == -2) and per-layer expert counts.

void xt_step
(
    at::Tensor sel,         // (picks) int64 global expert ids
    int64_t first,          // first global id of this rank's slice
    int64_t li,             // layer index in the pool
    int64_t E,              // local experts in this layer
    int64_t EM,             // row width of slotof / home (max local experts)
    int64_t S,              // slots
    at::Tensor slotof,      // (L, EM) int32: -2 static, -1 cold in host memory, >= 0 slot
    at::Tensor home,        // (L, EM, 3) int64 host addresses of gate / up / down trellis
    at::Tensor tabs,        // (L, 4) int64 addresses of the gate / up / down / (gate, up) pointer tables
    at::Tensor owner,       // (S) int32: li * EM + e, or -1
    at::Tensor refb,        // (S) int32 CLOCK reference bits
    at::Tensor pin,         // (S) int32 epoch of the last call that used the slot
    at::Tensor ctl,         // (4) int32: hand, epoch, jobs queued
    at::Tensor jobs,        // (2 * max jobs) int32: expert, slot
    at::Tensor stats,       // (8) int64: hits, misses, admitted, no victim, evictions
    int64_t arena,          // slot arena base address
    int64_t rec,            // slot size
    int64_t off_u,
    int64_t off_d
);

void xt_copy
(
    at::Tensor ctl,
    at::Tensor jobs,
    int64_t li,
    int64_t EM,
    at::Tensor home,
    int64_t arena,
    int64_t rec,
    int64_t sg,
    int64_t su,
    int64_t sd,
    int64_t off_u,
    int64_t off_d,
    int64_t grid
);
