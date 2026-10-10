#!/usr/bin/env python3
"""Residual stream of GLM-5.3-Flash after every block at the last prompt token, for harmful and harmless instruction
sets rendered by the model's own chat template in several contexts. The model runs in layer-split mode (no -tp: the
recording lives in this process), with exllamav3.util.cvec recording.

--prompts is a directory with one JSON list of strings per set: harmful-train.json, harmless-train.json,
harmful-val.json, harmless-val.json, harmful-heldout-jbb.json. I used AdvBench (train / val split), Alpaca
instructions as the harmless side and JailbreakBench as the held-out set. The prompt files are not in this repository.

    resid_dump.py --prompts DIR --out DIR [--limit N] <model_init args>

Contexts:  N  thinking off                  T  thinking on
           Ns thinking off, short system    Ta thinking on, coding-agent system    Na thinking off, coding-agent system
Output: OUT/resid/<set>.<ctx>.npy, float16 [prompts, blocks, streams, hidden], and <set>.<ctx>.json (shape, token counts).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from exllamav3 import model_init
from exllamav3.util import cvec

SIMPLE = "You are a helpful assistant."
AGENT = ("You are a coding agent working in a terminal on the user's machine. You can read and edit files and run shell "
         "commands through the provided tools. Keep answers short, explain what you changed and why, and ask before "
         "doing anything destructive. Follow the conventions of the repository you are in.")
CTX = {"N": (False, None), "T": (True, None), "Ns": (False, SIMPLE), "Ta": (True, AGENT), "Na": (False, AGENT)}
SETS = {"harmful-train": list(CTX), "harmless-train": list(CTX), "harmful-val": ["N", "T"], "harmless-val": ["N", "T"],
        "harmful-heldout-jbb": ["N", "T"]}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(ap, cache=False)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert not cvec.ENABLED, "record with EXL3_CVEC unset"

    t0 = time.time()
    model, config, _, tok = model_init.init(args, override_dynamic_seq_len=2048)
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    print(f" -- loaded in {time.time() - t0:.1f} s", flush=True)

    def render(instr, think, system):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": instr}]
        p = hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        if not think and p.endswith("<think>"):
            p += "</think>"                 # as the server does: thinking off = an empty think block
        assert p.endswith("<think>" if think else "<think></think>"), p[-40:]
        return p

    out = os.path.join(args.out, "resid"); os.makedirs(out, exist_ok=True)
    for name, ctxs in SETS.items():
        items = json.load(open(os.path.join(args.prompts, f"{name}.json")))
        if args.limit:
            items = items[:args.limit]
        for c in ctxs:
            path = os.path.join(out, f"{name}.{c}.npy")
            if os.path.exists(path):
                print(f"{name}.{c}: exists, skipped", flush=True); continue
            think, system = CTX[c]
            rows, ntok, t1 = [], [], time.time()
            for instr in items:
                ids = tok.encode(render(instr, think, system), encode_special_tokens=True)
                cvec.capture_begin()
                model.forward(ids, {"attn_mode": "flash_attn_nc", "last_tokens_only": 1})
                r = cvec.capture_end()
                assert r is not None, "nothing recorded: the forward did not run in this process"
                rows.append(r[:, 0].half().numpy()); ntok.append(int(ids.shape[-1]))
            a = np.stack(rows)
            np.save(path + ".tmp.npy", a); os.replace(path + ".tmp.npy", path)
            json.dump({"n_prompts": a.shape[0], "blocks": a.shape[1], "streams": a.shape[2], "hidden": a.shape[3],
                       "context": c, "thinking": think, "system": system, "tokens": ntok},
                      open(os.path.join(out, f"{name}.{c}.json"), "w"))
            print(f"{name}.{c}: {a.shape}, mean {np.mean(ntok):.0f} tokens, {(time.time() - t1) / len(items):.2f} s/prompt", flush=True)
    print("DUMP DONE", flush=True)
    try:
        model.unload()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
