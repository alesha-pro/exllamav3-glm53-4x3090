#!/usr/bin/env python3
"""Residual stream at the END of the model's own thinking (the </think> token), stock model, layer-split load.

    think_capture.py --prompts DIR --out DIR [--n 96] [--max-think 900] <model_init args>

The last prompt token says what the model is about to think; with thinking on, the decision to refuse is made while it
reasons. So: let the stock model think about harmful-train and harmless-train prompts, keep the runs whose thinking
closed, and record the stream after every block at the closing </think> token. Written as context "E" next to the
resid_dump.py files (OUT/resid/<set>.E.npy, [prompts, blocks, streams, hidden]); the thinking texts stay in OUT/think/.
"""
import argparse, json, os, time
import numpy as np
import torch
from exllamav3 import model_init, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.util import cvec


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(ap, cache=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--max-think", type=int, default=900)
    args = ap.parse_args()
    assert not cvec.ENABLED, "record with EXL3_CVEC unset"
    model, config, cache, tok, *_ = model_init.init(args)
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    gen = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=8, max_chunk_size=args.chunk_size)
    eos = set(int(e) for e in (getattr(config, "eos_token_id_list", None) or []))
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    os.makedirs(os.path.join(args.out, "think"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "resid"), exist_ok=True)

    def render(instr):
        p = hf_tok.apply_chat_template([{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        assert p.endswith("<think>"), p[-30:]
        return p

    for name in ("harmful-train", "harmless-train"):
        items = json.load(open(os.path.join(args.prompts, f"{name}.json")))[:args.n]
        texts = [""] * len(items)
        t0 = time.time()
        for i, p in enumerate(items):
            gen.enqueue(Job(input_ids=tok.encode(render(p), encode_special_tokens=True), max_new_tokens=args.max_think,
                            sampler=ArgmaxSampler(), stop_conditions=list(eos) + ["</think>"], decode_special_tokens=True, identifier=i))
        closed = [False] * len(items)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") != "streaming":
                    continue
                texts[r["identifier"]] += r.get("text") or ""
                if r.get("eos") and r.get("eos_reason") == "stop_string":
                    closed[r["identifier"]] = True
        keep = [i for i in range(len(items)) if closed[i] or "</think>" in texts[i]]
        json.dump([dict(prompt=items[i], thinking=texts[i].split("</think>")[0], closed=i in keep) for i in range(len(items))],
                  open(os.path.join(args.out, "think", f"{name}.json"), "w"), indent=1, ensure_ascii=False)
        print(f"{name}: thinking closed in {len(keep)}/{len(items)} within {args.max_think} tokens, {time.time() - t0:.0f} s", flush=True)
        rows, ntok = [], []
        for i in keep:
            full = render(items[i]) + texts[i].split("</think>")[0] + "</think>"
            ids = tok.encode(full, encode_special_tokens=True)
            cvec.capture_begin()
            model.forward(ids, {"attn_mode": "flash_attn_nc", "last_tokens_only": 1})
            r = cvec.capture_end()
            rows.append(r[:, 0].half().numpy()); ntok.append(int(ids.shape[-1]))
        a = np.stack(rows)
        np.save(os.path.join(args.out, "resid", f"{name}.E.npy"), a)
        json.dump({"n_prompts": a.shape[0], "blocks": a.shape[1], "streams": a.shape[2], "hidden": a.shape[3], "context": "E",
                   "note": "stream at the closing </think> of the stock model's own thinking", "tokens": ntok},
                  open(os.path.join(args.out, "resid", f"{name}.E.json"), "w"))
        print(f"{name}.E: {a.shape}, mean {np.mean(ntok):.0f} tokens", flush=True)
    print("THINK CAPTURE DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
