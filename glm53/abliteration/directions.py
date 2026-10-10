#!/usr/bin/env python3
"""Refusal directions of GLM-5.3-Flash measured on the quantized model itself, from resid_dump.py files (the wide
residual: 4 hyper-connection streams per block), and control-vector files for exllamav3.util.cvec built from them.

    directions.py ROOT analyze
    directions.py ROOT write NAME CTXS [A-B]      # ROOT/vectors/NAME.safetensors from contexts CTXS (e.g. N+T+Ns+Ta+Na),
                                                  # blocks A..B only (default: all but block 0)
    directions.py ROOT single NAME CTXS BLOCK     # block BLOCK's direction for every block (0..last), plus NAME.f32:
                                                  # the raw float32 vector Project Maya's STRATA_GLM_ABLATE reads

Per block, context and stream: d = mean(harmful) - mean(harmless) at the last prompt token (train split), normalized.
The unit directions are averaged over streams and contexts (every block writes the same output into all streams, so
one direction per block serves all of them). Then the components along the mean harmless state of every stream and
context of that block are removed: projecting the plain difference of means out also moves ordinary prompts, and with
projection that breaks the model (empty answers). The result is normalized.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

ALL = ["N", "T", "Ns", "Ta", "Na"]


def load(R, name, ctx):
    return np.load(R / f"resid/{name}.{ctx}.npy").astype(np.float32)          # [prompts, blocks, streams, hidden]


def means(R, ctx):
    return load(R, "harmful-train", ctx).mean(0, dtype=np.float64), load(R, "harmless-train", ctx).mean(0, dtype=np.float64)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def directions(R, ctxs, ortho=True):
    """[blocks, hidden] unit direction per block from the given contexts"""
    stats = {c: means(R, c) for c in ctxs}
    v = unit(np.mean([unit(mh - ms).mean(1) for mh, ms in stats.values()], axis=0))          # streams, then contexts
    if ortho:
        for l in range(v.shape[0]):
            basis = np.stack([ms[l, s] for _, ms in stats.values() for s in range(ms.shape[1])])   # harmless means
            q, _ = np.linalg.qr(basis.T)
            v[l] = v[l] - q @ (q.T @ v[l])
        v = unit(v)
    return v


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); r = x.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def analyze(R):
    out = {}
    per = {c: directions(R, [c]) for c in ALL}
    pooled = directions(R, ALL)
    plain = directions(R, ALL, ortho=False)
    L = pooled.shape[0]
    print("== per block: cos of each context's direction to the pooled one, cos(N, T), cos(pooled, previous block)")
    print("block  " + "  ".join(f"{c:>5}" for c in ALL) + "   cos(N,T)  cos(prev)")
    for l in range(0, L, 3):
        print(f"{l:5d}  " + "  ".join(f"{per[c][l] @ pooled[l]:5.2f}" for c in ALL) + f"   {per['N'][l] @ per['T'][l]:8.3f}  "
              f"{(pooled[l] @ pooled[l - 1]) if l else 0:8.3f}")
    out["cos_N_T"] = [float(per["N"][l] @ per["T"][l]) for l in range(L)]
    print("== stream agreement and size, context N: cos between the 4 per-stream directions (min, mean), |d| / |harmless mean| per stream")
    mh, ms = means(R, "N"); d = mh - ms; u = unit(d)
    S = d.shape[1]
    for l in range(0, L, 4):
        c = [float(u[l, a] @ u[l, b]) for a in range(S) for b in range(a + 1, S)]
        print(f"{l:5d}  min {min(c):.2f} mean {np.mean(c):.2f}   rel " + " ".join(f"{np.linalg.norm(d[l, s]) / np.linalg.norm(ms[l, s]):.2f}" for s in range(S)))
    print("== validation split, projection on the pooled direction averaged over streams: AUROC harmful vs harmless")
    print("block   val:N   val:T   jbb:N   jbb:T    harmless share of its norm along the direction (N): orthogonalized / plain")
    val = {(n, c): load(R, n, c) for n in ("harmful-val", "harmless-val", "harmful-heldout-jbb") for c in ("N", "T")}
    rows = []
    for l in range(1, L, 2):
        row = dict(block=l)
        for c in ("N", "T"):
            ph, ps, pj = (val[(n, c)][:, l].astype(np.float64) @ pooled[l] for n in ("harmful-val", "harmless-val", "harmful-heldout-jbb"))
            row[f"val_{c}"] = float(auroc(ph.mean(1), ps.mean(1))); row[f"jbb_{c}"] = float(auroc(pj.mean(1), ps.mean(1)))
        s = val[("harmless-val", "N")][:, l].astype(np.float64)
        row["share_ortho"] = float((np.abs(s @ pooled[l]) / np.linalg.norm(s, axis=-1)).mean())
        row["share_plain"] = float((np.abs(s @ plain[l]) / np.linalg.norm(s, axis=-1)).mean())
        rows.append(row)
        print(f"{l:5d}   {row['val_N']:.3f}   {row['val_T']:.3f}   {row['jbb_N']:.3f}   {row['jbb_T']:.3f}    {row['share_ortho']:.3f} / {row['share_plain']:.3f}")
    out["val"] = rows
    (R / "directions-analysis.json").write_text(json.dumps(out, indent=1) + "\n")


def write(R, name, ctxs, layers=None):
    import torch
    from safetensors.torch import save_file
    v = directions(R, ctxs.split("+"))
    L = v.shape[0]
    a, b = map(int, layers.split("-")) if layers else (1, L - 1)
    out = R / "vectors"; out.mkdir(exist_ok=True); p = out / f"{name}.safetensors"
    save_file({f"direction.{l}": torch.from_numpy(v[l].astype(np.float32)) for l in range(a, b + 1)}, str(p),
              metadata={"model": "GLM-5.3-Flash", "contexts": ctxs, "blocks": f"{a}-{b}", "recipe": "mean-difference, harmless-orthogonalized"})
    man = dict(file=str(p), contexts=ctxs, blocks=[a, b], sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
    (out / f"{name}.json").write_text(json.dumps(man, indent=1) + "\n"); print(json.dumps(man))


def single(R, name, ctxs, block):
    import torch
    from safetensors.torch import save_file
    v = directions(R, ctxs.split("+"))
    L, b = v.shape[0], int(block)
    d = v[b].astype(np.float32)
    out = R / "vectors"; out.mkdir(exist_ok=True); p = out / f"{name}.safetensors"
    save_file({f"direction.{l}": torch.from_numpy(d.copy()) for l in range(L)}, str(p),
              metadata={"model": "GLM-5.3-Flash", "contexts": ctxs, "blocks": f"0-{L - 1}", "recipe": f"block {b} direction on every block, mean-difference, harmless-orthogonalized"})
    (out / f"{name}.f32").write_bytes(d.tobytes())
    man = dict(file=str(p), raw=str(out / f"{name}.f32"), contexts=ctxs, source_block=b, blocks=[0, L - 1], norm=float(np.linalg.norm(d)),
               sha256=hashlib.sha256(p.read_bytes()).hexdigest(), sha256_f32=hashlib.sha256(d.tobytes()).hexdigest())
    (out / f"{name}.json").write_text(json.dumps(man, indent=1) + "\n"); print(json.dumps(man))


def basis(R, groups):
    """[blocks, k, hidden]: one direction per context group (e.g. "N+Ns+Na,T+Ta"), orthonormalized per block"""
    vs = np.stack([directions(R, g.split("+")) for g in groups.split(",")], axis=1)      # [blocks, k, hidden]
    out = np.zeros_like(vs)
    for l in range(vs.shape[0]):
        q, _ = np.linalg.qr(vs[l].T)
        out[l] = q.T
    return out


def multi(R, name, groups, spec=None):
    """multi NAME GROUPS [A-B | single:B]: several directions per block; single:B = block B's set on every block, plus NAME.f32"""
    import torch
    from safetensors.torch import save_file
    v = basis(R, groups).astype(np.float32)
    L = v.shape[0]
    out = R / "vectors"; out.mkdir(exist_ok=True); p = out / f"{name}.safetensors"
    if spec and spec.startswith("single:"):
        b = int(spec[7:])
        save_file({f"direction.{l}": torch.from_numpy(v[b].copy()) for l in range(L)}, str(p))
        (out / f"{name}.f32").write_bytes(v[b].tobytes())
        man = dict(file=str(p), raw=str(out / f"{name}.f32"), groups=groups, source_block=b, blocks=[0, L - 1], k=int(v.shape[1]))
    else:
        a, b = map(int, spec.split("-")) if spec else (1, L - 1)
        save_file({f"direction.{l}": torch.from_numpy(v[l].copy()) for l in range(a, b + 1)}, str(p))
        man = dict(file=str(p), groups=groups, blocks=[a, b], k=int(v.shape[1]))
    man["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / f"{name}.json").write_text(json.dumps(man, indent=1) + "\n"); print(json.dumps(man))


if __name__ == "__main__":
    R = Path(sys.argv[1])
    cmd = sys.argv[2]
    analyze(R) if cmd == "analyze" else single(R, *sys.argv[3:6]) if cmd == "single" else multi(R, *sys.argv[3:6]) if cmd == "multi" \
        else write(R, *sys.argv[3:6])
