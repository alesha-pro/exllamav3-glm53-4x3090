"""
Expert tiering for expert-parallel MoE layers under tensor parallelism (experimental).

A model whose routed experts do not fit in VRAM keeps the least-used experts of every rank in
page-locked host memory instead: their EXL3 trellis tensors are laid out in one pinned arena per
(layer, rank) and the modules get a zero-copy device alias (the mechanism of pinned vision
towers), so every kernel reads them over PCIe through the same pointer tables as resident
experts. The arithmetic is unchanged; only the address of a cold expert's bytes differs.

Placement is static: each rank ranks the experts of its own slices (all layers) by a usage
score and keeps the top EXL3_XTIER_HOT_FRAC of them in VRAM, so every rank holds the same
number of resident experts whatever the per-layer skew.

EXL3_XTIER_STATS      JSON {module key: [score per expert]} (e.g. routing counts); enables tiering
EXL3_XTIER_HOT_FRAC   fraction of each rank's experts kept in VRAM (default 1.0 = all, tiering off)
EXL3_XTIER_SLOTS      per rank: VRAM slots of a device-side CLOCK cache for the cold experts (default 0:
                      cold experts are always read from host memory). A cold expert picked at decode is
                      copied into a slot once and stays there until evicted (see exllamav3_ext/xtier.cuh)
EXL3_XTIER_MASK       1: the router never selects a cold expert (its selection bias is pushed far down on
                      every rank, so the next best resident expert takes the slot). Nothing is read from
                      host memory any more. This changes the model's output: it is expert pruning by
                      usage statistics, an experiment to price the misses, not a lossless mode
"""
import os, json
import torch

_STATS_PATH = os.environ.get("EXL3_XTIER_STATS")
HOT_FRAC = float(os.environ.get("EXL3_XTIER_HOT_FRAC", "1.0"))
MASK = os.environ.get("EXL3_XTIER_MASK", "0") not in ("", "0")
MASK_VALUE = -1.0e4     # selection bias of a masked expert (scores are below 1, real biases a few units)
_stats = None
_cold_cache = {}    # id(plan) -> {key: frozenset(cold global expert ids)}
SUMMARY = {"layers": 0, "cold_experts": 0, "pinned_bytes": 0}


def enabled() -> bool:
    return bool(_STATS_PATH) and HOT_FRAC < 1.0


def stats() -> dict:
    global _stats
    if _stats is None:
        with open(_STATS_PATH) as f:
            _stats = json.load(f)
    return _stats


def hot_fraction(key: str) -> float:
    """Share of a layer's expert storage that stays in VRAM, for the split planner"""
    return HOT_FRAC if enabled() and key in stats() else 1.0


def cold_set(plan: dict, key: str) -> frozenset:
    """Global expert ids of layer `key` that this rank (whose plan dict is given) keeps in host memory"""
    if not enabled():
        return frozenset()
    c = _cold_cache.get(id(plan))
    if c is None:
        st = stats()
        cand = []
        for k, v in plan.items():
            if k in st and len(v) == 3 and v[2] == "experts":
                sc = st[k]
                cand += [(sc[e], k, e) for e in range(v[0], v[1])]
        cand.sort(key = lambda t: -t[0])
        n_hot = int(round(len(cand) * HOT_FRAC))
        c = {}
        for _, k, e in cand[n_hot:]:
            c.setdefault(k, set()).add(e)
        c = {k: frozenset(s) for k, s in c.items()}
        _cold_cache[id(plan)] = c
        _cold_cache[("keep", id(plan))] = plan   # keep the id stable
    return c.get(key, frozenset())


def cold_set_all(plans: dict, key: str) -> frozenset:
    """Global expert ids of layer `key` that any rank keeps out of VRAM (plans: the whole TP plan, per device)"""
    out = set()
    for plan in (plans.values() if isinstance(plans, dict) else plans):
        if isinstance(plan, dict):
            out |= cold_set(plan, key)
    return frozenset(out)


def desc_nbytes(d: dict) -> int:
    """Byte size of a tensor behind a TP producer descriptor"""
    if d is None or d.get("method") == "none_tensor":
        return 0
    if d.get("method") == "share_memory":
        t = d["shared_tensor"]
        return t.numel() * t.element_size()
    return int(d["nbytes"])


class Arena:
    """One pinned, device-mapped region holding the trellis tensors of a layer's cold experts"""

    def __init__(self, nbytes: int, what: str, device: int):
        from .pinned_arena import PinnedArena
        self.arena = PinnedArena(nbytes, what, mapped = True, required = True)
        self.device = device
        self.off = 0
        SUMMARY["layers"] += 1
        SUMMARY["pinned_bytes"] += self.arena.size

    def put(self, t: torch.Tensor):
        """Copy a CPU tensor into the arena; returns (device alias, host tensor to keep alive)"""
        from ..ext import exllamav3_ext as ext
        nbytes = t.numel() * t.element_size()
        off = (self.off + 255) // 256 * 256
        assert off + nbytes <= self.arena.size, "xtier arena overflow"
        p = self.arena.tensor[off: off + nbytes].view(t.dtype).view(t.shape)
        p.copy_(t)
        self.off = off + nbytes
        return ext.pinned_cuda_view(p, self.device), p


# ---- device-side cache of cold experts -----------------------------------------------------------

SLOTS = int(os.environ.get("EXL3_XTIER_SLOTS", "0"))
_ADMIT_MAX = int(os.environ.get("EXL3_XTIER_ADMIT_MAX", "64"))
REGISTRY = []       # tiered BlockSparseMLP modules of this process, in import order
POOL = None
POOLS = []


def register(module, cold_local: list):
    """Called by BlockSparseMLP.tp_import for a layer with cold experts (local expert indices)"""
    module._xtier_cold = cold_local
    REGISTRY.append(module)


class Pool:
    def __init__(self, mods: list, slots: int):
        from ..ext import exllamav3_ext as ext
        self.ext = ext
        self.mods = mods
        m0 = mods[0]
        device = torch.device(m0.device) if not isinstance(m0.device, int) else torch.device("cuda", m0.device)
        self.device = device
        nb = lambda l: l.inner.trellis.numel() * l.inner.trellis.element_size()
        gated = m0.gated
        self.sg = nb(m0.multi_gate.linears[0]) if gated else 0
        self.su = nb(m0.multi_up.linears[0])
        self.sd = nb(m0.multi_down.linears[0])
        al = lambda x: (x + 255) // 256 * 256
        self.off_u = al(self.sg)
        self.off_d = self.off_u + al(self.su)
        self.rec = self.off_d + al(self.sd)
        assert self.sg % 16 == 0 and self.su % 16 == 0 and self.sd % 16 == 0
        L = len(mods)
        EM = max(m.num_local_experts for m in mods)
        self.EM = EM
        self.S = slots
        self.arena = torch.empty((max(slots, 1), self.rec), dtype = torch.uint8, device = device)
        i32 = dict(dtype = torch.int32, device = device)
        slotof = torch.full((L, EM), -2, dtype = torch.int32)
        home = torch.zeros((L, EM, 3), dtype = torch.int64)
        tabs = []
        self.keep = []
        for li, m in enumerate(mods):
            E = m.num_local_experts
            for mm, sz in (((m.multi_gate, self.sg),) if gated else ()) + ((m.multi_up, self.su), (m.multi_down, self.sd)):
                for l in mm.linears:
                    assert nb(l) == sz, "xtier: experts of the pool must share one trellis layout"
            pg = m.multi_gate.ptrs_trellis if gated else m.multi_up.ptrs_trellis
            pu, pd = m.multi_up.ptrs_trellis, m.multi_down.ptrs_trellis
            gu = getattr(m, "_gu_trellis_ptr", None)
            assert pg.dtype == torch.long and pg.is_contiguous() and pu.is_contiguous() and pd.is_contiguous()
            home[li, :E, 0] = pg.cpu(); home[li, :E, 1] = pu.cpu(); home[li, :E, 2] = pd.cpu()
            for e in m._xtier_cold:
                slotof[li, e] = -1
            tabs.append([pg.data_ptr(), pu.data_ptr(), pd.data_ptr(), gu.data_ptr() if gu is not None else 0])
            self.keep.append((pg, pu, pd, gu))
            m._xt = (self, li, E)
        self.slotof = slotof.to(device)
        self.home = home.to(device)
        self.tabs = torch.tensor(tabs, dtype = torch.int64, device = device)
        self.owner = torch.full((max(slots, 1),), -1, **i32)
        self.refb = torch.zeros((max(slots, 1),), **i32)
        self.pin = torch.zeros((max(slots, 1),), **i32)
        self.ctl = torch.zeros((4,), **i32)
        self.jobs = torch.zeros((2 * 1024,), **i32)
        self.stats = torch.zeros((8,), dtype = torch.int64, device = device)
        # Blocks of the copy kernel: launched on every tiered layer call (the device decides whether there
        # is anything to copy), so kept small; a few dozen blocks already saturate the PCIe link
        self.grid = int(os.environ.get("EXL3_XTIER_COPY_GRID", "32"))
        self.arena_ptr = self.arena.data_ptr()
        print(f" -- xtier: {device}: {L} tiered layers, {sum(len(m._xtier_cold) for m in mods)} cold experts, "
              f"{slots} cache slots x {self.rec / 1e6:.2f} MB = {slots * self.rec / 1e9:.2f} GB", flush = True)

    def step(self, li: int, E: int, first: int, sel: torch.Tensor):
        sel = sel.reshape(-1)
        if sel.dtype != torch.long or not sel.is_contiguous():
            sel = sel.long().contiguous()
        e = self.ext
        e.xt_step(sel, first, li, E, self.EM, self.S, self.slotof, self.home, self.tabs, self.owner, self.refb,
                  self.pin, self.ctl, self.jobs, self.stats, self.arena_ptr, self.rec, self.off_u, self.off_d)
        e.xt_copy(self.ctl, self.jobs, li, self.EM, self.home, self.arena_ptr, self.rec, self.sg, self.su, self.sd,
                  self.off_u, self.off_d, self.grid)

    def summary(self) -> dict:
        s = self.stats.tolist()
        return {"device": str(self.device), "slots": self.S, "hits": s[0], "misses": s[1], "admitted": s[2],
                "no_victim": s[3], "evictions": s[4], "hit_rate": round(s[0] / max(1, s[0] + s[1]), 4)}


def _layout(m) -> tuple:
    nb = lambda l: l.inner.trellis.numel() * l.inner.trellis.element_size()
    return (nb(m.multi_gate.linears[0]) if m.gated else 0, nb(m.multi_up.linears[0]), nb(m.multi_down.linears[0]))


def pool_step(module, sel: torch.Tensor):
    """Before the decode expert kernels of a tiered layer: account the picks and admit cold ones"""
    global POOL
    xt = getattr(module, "_xt", None)
    if xt is None:
        if POOL is not None or not REGISTRY:
            return
        # One pool per trellis layout: a mixed quant has layers with different expert sizes. The slot
        # budget is split by the number of cold experts in each group
        groups = {}
        for m in REGISTRY:
            groups.setdefault(_layout(m), []).append(m)
        total = max(1, sum(len(m._xtier_cold) for m in REGISTRY))
        for mods in groups.values():
            cold = sum(len(m._xtier_cold) for m in mods)
            POOLS.append(Pool(mods, max(8, round(SLOTS * cold / total)) if SLOTS else 0))
        POOL = POOLS[0]
        xt = module._xt
    if sel.numel() <= _ADMIT_MAX:
        xt[0].step(xt[1], xt[2], int(module.routing_first or 0), sel)


def print_summary(reset: bool = True):
    for pool in POOLS:
        print(f" -- xtier stats: {pool.summary()}", flush = True)
        if reset:
            with torch.inference_mode():
                pool.stats.zero_()
    return True
