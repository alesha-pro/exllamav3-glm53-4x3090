#!/usr/bin/env python3
"""KL divergence of a model as exllamav3 loads and runs it (any -tp / environment configuration) against stored
full-vocabulary teacher logits: the sealed 25-window panel of brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits.

Same definition as the fidelity table in the turboderp/GLM-5.3-Flash-exl3 discussion: teacher-forced, KLD(teacher ||
student) in nats over the full vocabulary, fp64 accumulation, 2047 prediction positions per 2048-token window, mean
over all positions; plus top-1 agreement. The difference: there the student ran through a transformers reference
forward, here through the engine itself, so the number includes whatever the serving path does (tensor-parallel wire
rounding, expert tiering).

  kld_eval.py --ref DIR [--windows 25] [--out FILE] [--tag NAME] <model_init args: -m ... -tp ...>
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from safetensors import safe_open
from exllamav3 import model_init


def kl_rows(t_cpu, s_dev, step=128):
    """t_cpu [n, V] fp32 on the host, s_dev [n, V] on a device -> per-position KL(t || s) (fp64, host), top-1 matches"""
    kls, agree = [], 0
    for i in range(0, t_cpu.shape[0], step):
        a = t_cpu[i:i + step].to(s_dev.device, torch.float64)
        b = s_dev[i:i + step].to(torch.float64)
        la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kls.append((la.exp() * (la - lb)).sum(-1).cpu())
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
    return torch.cat(kls), agree


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(ap, cache=False)
    ap.add_argument("--ref", required=True, help="dataset root: logits/window-NNNN.safetensors, calibration/panel-v1/arrays/final-NNNN.tokens.npy")
    ap.add_argument("--windows", type=int, default=25)
    ap.add_argument("--out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    model, config, _, tok = model_init.init(args, override_dynamic_seq_len=2048, max_output_size=2048, max_output_factor=5)
    print(f" -- loaded in {time.time() - t0:.1f} s", flush=True)

    all_kl, agree, npos, per_window, offset = [], 0, 0, [], None
    for w in range(args.windows):
        ids = np.load(os.path.join(args.ref, f"calibration/panel-v1/arrays/final-{w:04d}.tokens.npy"), allow_pickle=False)
        assert ids.shape == (2048,), ids.shape
        with safe_open(os.path.join(args.ref, f"logits/window-{w:04d}.safetensors"), "pt") as f:
            keys = list(f.keys())
            key = "logits" if "logits" in keys else max(keys, key=lambda k: f.get_slice(k).get_shape()[-1])
            teacher = f.get_tensor(key)
        if w == 0:
            print(f" -- teacher window 0: keys {keys}, '{key}' {tuple(teacher.shape)} {teacher.dtype}", flush=True)
        teacher = teacher.reshape(-1, teacher.shape[-1]).float()
        n, V = teacher.shape
        logits = model.forward(torch.from_numpy(ids.astype(np.int64))[None], {"attn_mode": "flash_attn_nc"})
        student = logits[0, :, :V]
        if w == 0:
            print(f" -- student logits {tuple(logits.shape)} {logits.dtype} on {logits.device}", flush=True)
            # which student rows the teacher rows are: try both alignments once, the right one is far smaller
            cand = {}
            for off in ((0, 1) if n < student.shape[0] else (0,)):
                k, _ = kl_rows(teacher[:256], student[off:off + 256])
                cand[off] = float(k.mean())
            offset = min(cand, key=cand.get)
            print(f" -- alignment check, mean KL over 256 rows per row offset: {cand} -> offset {offset}", flush=True)
        kl, a = kl_rows(teacher, student[offset:offset + n])
        all_kl.append(kl); agree += a; npos += n
        per_window.append({"window": w, "mean_kld": float(kl.mean()), "top1": a / n})
        print(f"window {w:2d}: mean KLD {float(kl.mean()):.6f}, top-1 {a / n:.4f}", flush=True)
        del logits, student, teacher
        torch.cuda.empty_cache()

    kl = torch.cat(all_kl)
    q = lambda p: float(torch.quantile(kl, p))
    res = {"tag": args.tag, "argv": sys.argv[1:], "env": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
           "windows": args.windows, "positions": npos, "row_offset": offset,
           "mean_kld": float(kl.mean()), "median_kld": q(0.5), "p90_kld": q(0.9), "p99_kld": q(0.99), "max_kld": float(kl.max()),
           "top1_agreement": agree / npos, "per_window": per_window, "elapsed_s": round(time.time() - t0, 1)}
    print("RESULT " + json.dumps({k: res[k] for k in ("tag", "positions", "mean_kld", "median_kld", "p90_kld", "p99_kld", "top1_agreement")}), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    try:
        model.unload()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
