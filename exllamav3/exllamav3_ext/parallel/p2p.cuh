#pragma once

#include <ATen/Tensor.h>
#include <string>
#include <vector>

// Peer-to-peer collectives for the native tensor-parallel backend (decode-size payloads).
//
// Each rank owns a mailbox in its own device memory and exports it over CUDA IPC; every other rank maps it
// with lazy peer access and writes into it straight from its kernels (GPU -> GPU over PCIe, no host memory
// and no CPU helper in the path). A reduce is one kernel per rank: write the local contribution and a stage
// flag into every peer's mailbox, spin on the stage flags in the own mailbox (local device reads), then sum
// the contributions in rank order. Every rank adds the same fp32 values in the same order, so the result is
// bit-identical across ranks. With a single contributor the payload is copied verbatim (a broadcast).
//
// Mailbox layout (ndev = highest device index + 1):
//   [0, 64)                                   own stage counter (u32)
//   [64 + (src * NSLOT + slot) * 64]          stage flag written by rank src (u32; bit 31: no contribution)
//   [data_off + (src * NSLOT + slot) * SLOT]  payload written by rank src
#define P2P_NSLOT 4
#define P2P_SLOT_BYTES (512 * 1024)

size_t pg_p2p_mailbox_size(int ndev);

// Allocate this rank's mailbox (raw device allocation, zeroed). handle_out receives the IPC handle bytes
uintptr_t pg_p2p_create(int device, int ndev, std::string& handle_out);

// Map a peer's mailbox into this rank's address space; returns the pointer usable in this rank's kernels
uintptr_t pg_p2p_open(int device, const std::string& handle);

// Whether a tensor can take the p2p path (size and alignment)
bool pg_p2p_fits(const at::Tensor& tensor);

void pg_all_reduce_p2p
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t mailbox,
    std::vector<uintptr_t> peers,       // indexed by device, 0 where unused
    at::Tensor& abort_flag
);
