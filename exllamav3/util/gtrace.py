"""
Device-side forward-pass tracer (debug, off by default).

EXL3_GTRACE=1 records a CUDA timing event after every module of a forward pass, plus any finer
marks the modules drop with mark(), and aggregates the device-side time between consecutive
marks per label. Events are queued on the stream and read back at the start of the next pass,
so tracing adds a few microseconds of host time per mark and no synchronization. The host-side
time between the same marks is kept alongside: a segment whose host time is close to its device
time is host-bound (the device waited for the launch), one whose host time is far smaller is
device-bound.

EXL3_GTRACE_OUT    path prefix for the per-process JSON dumps (default /tmp/exl3_gtrace)
EXL3_GTRACE_EVERY  rewrite the dump every N traced passes (default 100)
EXL3_GTRACE_ROWS   only trace passes with at most this many rows (default 16: decode shapes)

Each rank of a tensor-parallel model traces its own module walk and writes its own file.
"""
import os, time, json
import torch

ENABLED = os.environ.get("EXL3_GTRACE", "0") not in ("", "0")
# EXL3_GTRACE=2: light mode, only the pass begin / end marks (two events per pass): the host
# time of the whole module walk against the device time of the pass, at negligible overhead
LIGHT = os.environ.get("EXL3_GTRACE", "0") == "2"
# EXL3_GTRACE_PREFILL=1: also trace the prefill passes of a tensor-parallel model (raise EXL3_GTRACE_ROWS to
# the chunk size as well)
PREFILL = os.environ.get("EXL3_GTRACE_PREFILL", "0") not in ("", "0")
_OUT = os.environ.get("EXL3_GTRACE_OUT", "/tmp/exl3_gtrace")
_EVERY = int(os.environ.get("EXL3_GTRACE_EVERY", "100"))
_ROWS = int(os.environ.get("EXL3_GTRACE_ROWS", "16"))

_cur = None        # marks of the pass being recorded: list of (label, device index, event, host time)
_cur_key = None
_pending = None    # (group key, marks) of the previous pass, read back at the next begin()
_groups = {}       # "tag.rN" -> {"agg": {label: [device ms, host ms, count]}, "passes": n, "dev": ms, "host": ms}
_total = 0
_rows_hist = {}
_proc = None
_pool = []         # recycled events
_streams = {}      # device index -> its stream (the module walks stay on the default stream)


def _dev_index(device):
    if device is None:
        return None
    if isinstance(device, int):
        return device if device >= 0 else None
    d = torch.device(device)
    if d.type != "cuda":
        return None
    return d.index if d.index is not None else torch.cuda.current_device()


def begin(rows: int, tag, device):
    """Start tracing a pass (rows = bsz * seq_len of the input). Reads back the previous pass."""
    global _cur, _cur_key, _proc
    if not ENABLED:
        return
    flush()
    if _proc is None:
        _proc = str(tag).split(".n")[0]
    k = f"{tag}.r{rows}"
    _rows_hist[k] = _rows_hist.get(k, 0) + 1
    if rows > _ROWS:
        _cur = None
        return
    _cur_key = k
    _cur = []
    mark("_begin", device)


def mark(label: str, device = None):
    """Record a mark on the given device's current stream (default: the last marked device)."""
    if _cur is None or (LIGHT and label[0] != "_"):
        return
    di = _dev_index(device)
    if di is None:
        di = _cur[-1][1] if _cur else None
        if di is None:
            return
    ev = _pool.pop() if _pool else torch.cuda.Event(enable_timing = True)
    st = _streams.get(di)
    if st is None:
        st = _streams[di] = torch.cuda.current_stream(di)
    ev.record(st)
    _cur.append((label, di, ev, time.perf_counter()))


def end():
    global _cur, _pending
    if _cur is None:
        return
    _pending = (_cur_key, _cur)
    _cur = None


def flush():
    """Fold the finished pass into the aggregate (waits for its last event if still running)."""
    global _pending, _total
    if not _pending:
        return
    key, p = _pending
    _pending = None
    if not p[-1][2].query():
        p[-1][2].synchronize()
    g = _groups.get(key)
    if g is None:
        g = _groups[key] = {"agg": {}, "passes": 0, "dev": 0.0, "host": 0.0}
    agg = g["agg"]
    tot_d = tot_h = 0.0
    for (l0, d0, e0, h0), (l1, d1, e1, h1) in zip(p, p[1:]):
        host = (h1 - h0) * 1e3
        if d0 == d1:
            dev = e0.elapsed_time(e1)
        else:
            dev = host   # device hop: events of two devices share no clock, the copy is synchronous
            l1 = l1 + "@hop"
        a = agg.get(l1)
        if a is None:
            a = agg[l1] = [0.0, 0.0, 0]
        a[0] += dev; a[1] += host; a[2] += 1
        tot_d += dev; tot_h += host
    g["dev"] += tot_d; g["host"] += tot_h; g["passes"] += 1
    _pool.extend(m[2] for m in p)
    _total += 1
    if _total % _EVERY == 0:
        dump()


def dump():
    if not _total:
        return
    out = {"proc": _proc, "pid": os.getpid(), "rows_hist": _rows_hist, "groups": {}}
    for key, g in _groups.items():
        n = g["passes"]
        out["groups"][key] = {
            "passes": n, "pass_dev_ms": g["dev"] / n, "pass_host_ms": g["host"] / n,
            "labels": {k: {"dev_ms_per_pass": v[0] / n, "host_ms_per_pass": v[1] / n, "n_per_pass": v[2] / n}
                       for k, v in g["agg"].items()},
        }
    path = f"{_OUT}.{_proc}.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)


def reset():
    global _total, _pending
    _groups.clear(); _rows_hist.clear()
    _total = 0
    _pending = None


def checkpoint(name: str):
    """Write what was collected so far to its own file (suffix name) and start over: one process, several
    measured variants"""
    if not ENABLED:
        return False
    flush()
    if _total:
        global _OUT
        keep = _OUT
        _OUT = f"{keep}.{name}"
        try:
            dump()
        finally:
            _OUT = keep
    reset()
    return True
