#pragma once

#include <ATen/Tensor.h>
#include <string>
#include <vector>

// Peer-to-peer all-reduce for prefill-size payloads (native tensor-parallel backend, 2 / 4 / 8 ranks).
//
// The CPU-assisted reduce streams every rank's contribution through pinned host memory in 128 KB stages
// and sums it on the host: about 1.2 GB/s per rank for a 32 MiB payload on a PCIe 3.0, two-channel DDR4
// host, which makes the collectives the largest part of a prefill pass. Here the ranks exchange pairwise
// instead, straight between device memories (CUDA IPC mailboxes with peer access, copy engine DMA at link
// speed, both directions of a pair at once), and sum on the device:
//
//   round j (j = 0 .. log2(ranks) - 1): rank i and rank i ^ (1 << j) send each other their running sum
//   (16-bit wire) and add what arrives. After the last round every rank holds the total.
//
// Addition is commutative, so the two ranks of a pair compute bit-identical sums and all ranks end with
// the same value: 4 ranks give (p0 + p1) + (p2 + p3) everywhere. The wire is 16 bits as on the CPU path
// (bf16 for fp32 / bf16 tensors, fp16 for fp16 tensors); a running sum is rounded to the wire once per
// round, the last add is delivered unrounded for fp32 tensors.
//
// Mailbox layout (slot = slot_bytes, rounds = log2(ranks)):
//   [0, 4096)                      header: u32 flag of round j at 64 * j, written by that round's partner
//                                  (value = stage + 1 once its payload has landed)
//   slot j, j < rounds             own wire buffer sent in round j (local scratch)
//   slot rounds + 2 * j + parity   inbox of round j, written by the partner (parity = stage & 1: a partner
//                                  cannot get two reduces ahead, so one spare slot makes overwrites impossible)
//
// Broadcast (a module one rank owns whole): the owner's tensor travels down a binomial tree over the same
// pairs, byte for byte and in its own dtype: in round j every rank that already has the data sends it to
// its round-j partner. The receiver's copy lands in a region of its own behind the reduce slots (a reduce
// of the next module may already be arriving while a rank still waits for its broadcast, so the two must
// not share memory) and a kernel moves it into the tensor. A sender does not wait for the receiver, so the
// caller must not issue two broadcasts without an all-reduce (any path) in between: an all-reduce only
// completes once every rank has contributed, which proves the receivers have drained the region.
//   header: u32 flag of broadcast round j at 1024 + 64 * j, written by the sender
//   [P2PB_HEADER + 3 * rounds * slot, + bcast_bytes)   broadcast inbox
#define P2PB_HEADER 4096
#define P2PB_BFLAG_OFF 1024
#define P2PB_MAX_ROUNDS 3

size_t pg_p2pb_mailbox_size(size_t slot_bytes, int rounds, size_t bcast_bytes);

// Allocate this rank's mailbox (raw device allocation, zeroed). handle_out receives the IPC handle bytes.
// Peers map it with pg_p2p_open
uintptr_t pg_p2pb_create(int device, size_t slot_bytes, int rounds, size_t bcast_bytes, std::string& handle_out);

// Whether a tensor takes this path: contiguous fp32 / fp16 / bf16, a multiple of 8 elements, wire size
// within [min_bytes, slot_bytes]
bool pg_p2pb_fits(const at::Tensor& tensor, size_t slot_bytes, size_t min_bytes);

// Whether a tensor can take the p2p broadcast: contiguous, a multiple of 16 bytes, within
// [min_bytes, bcast_bytes]
bool pg_p2pb_bcast_fits(const at::Tensor& tensor, size_t bcast_bytes, size_t min_bytes);

void pg_broadcast_p2p_big
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    int src_device,
    at::Tensor& tensor,
    uintptr_t mailbox,
    std::vector<uintptr_t> peers,       // indexed by device, 0 where unused
    size_t slot_bytes,
    size_t bcast_bytes,
    at::Tensor& abort_flag
);

void pg_all_reduce_p2p_big
(
    uintptr_t ctx,
    uintptr_t ctx_dev,
    std::vector<uintptr_t> devices,
    int this_device,
    at::Tensor& tensor,
    bool contributor,
    uintptr_t mailbox,
    std::vector<uintptr_t> peers,       // indexed by device, 0 where unused
    size_t slot_bytes,
    at::Tensor& abort_flag
);
