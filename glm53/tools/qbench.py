#!/usr/bin/env python3
"""Quick in-process speed probe for exllamav3 on GLM-5.3-Flash (stock or patched engine).

  prefill: N fresh random tokens, tok/s = N / time to first generated token
  decode : greedy chat completion (thinking off), fixed --gen tokens, tok/s = (n-1) / (t_last - t_first)
           optionally behind a random-token context prefix (--dec-ctx)

Not the campaign protocol (that is offload/bench/sweep.py against serve.py); this is the fast loop for A/B work.
"""
import argparse, json, os, random, sys, time
import torch

TOPICS = ["the history of the printing press", "how a jet engine works", "the causes of the French revolution",
          "photosynthesis in C4 plants", "how TCP congestion control works", "the life cycle of stars"]


def main():
    from exllamav3 import model_init, Generator, Job
    from exllamav3.generator.sampler import ArgmaxSampler
    ap = argparse.ArgumentParser()
    model_init.add_args(ap, cache=True, add_draft_model_args=True, default_chunk_size=2048)
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192])
    ap.add_argument("--gen", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dec-ctx", type=int, nargs="*", default=[0], help="random-token prefix lengths for decode runs")
    ap.add_argument("--conc", type=int, nargs="*", default=[1])
    ap.add_argument("--max-batch-size", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--tag", default="")
    ap.add_argument("--pre", help="python module path to import before model load (engine patches)")
    ap.add_argument("--show", action="store_true", help="print the first decoded answer")
    ap.add_argument("--ndt-sweep", type=int, nargs="*", default=None,
                    help="decode once per draft length (0 = no drafting); needs -mtp / a draft model for k > 0")
    ap.add_argument("--chunk-sweep", type=int, nargs="*", default=None, help="prefill once per generator chunk size")
    ap.add_argument("--natural", action="store_true", help="decode to natural EOS instead of --gen tokens")
    ap.add_argument("--pyspy", type=float, default=0, help="seconds of py-spy sampling of every rank during the last decode round")
    ap.add_argument("--nsys", help="nsys session name: collect only around the last decode round of each variant")
    ap.add_argument("--nsys-out", default="/tmp/qb_nsys")
    ap.add_argument("--fixed", action="store_true", help="fixed decode prompts (no nonce): outputs comparable across variants")
    ap.add_argument("--needle", type=int, nargs="*", default=None,
                    help="token lengths of a natural-text records document to query: correctness check of long prompts, and a "
                         "prefill rate on real text")
    ap.add_argument("--ab", nargs="*", default=None,
                    help="variants 'name|module:attr=value|...' applied in every rank before the decode runs")
    args = ap.parse_args()
    if args.pre:
        import importlib.util
        spec = importlib.util.spec_from_file_location("qb_pre", args.pre)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    t0 = time.time()
    model, config, cache, tok, draft_model, _dc, draft_cache = model_init.init(args)
    t_load = time.time() - t0
    vram = [round(torch.cuda.mem_get_info(i)[1] / 2**20 - torch.cuda.mem_get_info(i)[0] / 2**20) for i in range(torch.cuda.device_count())]
    print(f" -- loaded in {t_load:.1f} s, VRAM used MiB {vram}", flush=True)
    try:
        from exllamav3.util import abtest as _ab
        _ab.call_all([model], "exllamav3.util.memrep", "print_mem", "load")
        _ab.call_all([model], "exllamav3.util.memrep", "reset_peak")
    except Exception as e:
        print(f" -- mem report unavailable: {e}", flush=True)
    state = {"gen": None}

    def make_gen(ndt, chunk):
        if state["gen"] is not None:
            try:
                state["gen"].close()
            except Exception as e:
                print(f" !! generator close: {e}", flush=True)
        use_draft = draft_model is not None and ndt != 0
        state["gen"] = Generator(model=model, cache=cache, tokenizer=tok, max_batch_size=args.max_batch_size,
                                 max_chunk_size=chunk, draft_model=draft_model if use_draft else None,
                                 draft_cache=draft_cache if use_draft else None,
                                 num_draft_tokens=ndt if use_draft else None, dynamic_draft_tokens=args.dynamic_draft,
                                 draft_confidence=args.draft_confidence, ngram_match_min=args.ngram_match_min,
                                 record_draft_stats=use_draft,
                                 recurrent_cache_size=int(args.recurrent_cache_size * 1024 ** 3))
        return state["gen"]

    gen = make_gen(args.num_draft_tokens, args.chunk_size)
    eos = set(int(e) for e in (getattr(config, "eos_token_id_list", None) or []))
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    try:
        gc = json.load(open(os.path.join(args.model_dir, "generation_config.json")))
        e = gc.get("eos_token_id")
        for x in (e if isinstance(e, list) else [e]):
            if x is not None:
                eos.add(int(x))
    except Exception:
        pass
    res = {"tag": args.tag, "argv": sys.argv[1:], "load_s": round(t_load, 1), "vram_mib": vram, "prefill": {}, "decode": {},
           "env": {k: v for k, v in os.environ.items() if k.startswith(("EXL3_", "GLM53_", "EXLLAMA"))}}
    rng = random.Random(1234)
    seq = [0]

    def rand_ids(n):
        return [rng.randrange(1000, 150000) for _ in range(n)]

    def chat_ids(i):
        seq[0] += 1
        nonce = "" if args.fixed else f"[cache-bust nonce: {seq[0]}-{rng.randrange(1 << 30)}]\n"
        q = f"{nonce}Write a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} (variant {i})."
        return tok.encode(f"[gMASK]<sop><|user|>{q}<|assistant|><think></think>", encode_special_tokens=True)[0].tolist()

    def run(jobs_ids, max_new, stop):
        """Run jobs together; returns per-job dict(t0, first, last, n, text, draft)."""
        jobs = []
        for ids in jobs_ids:
            j = Job(input_ids=torch.tensor([ids], dtype=torch.long), max_new_tokens=max_new, sampler=ArgmaxSampler(),
                    stop_conditions=list(eos) if stop else [], decode_special_tokens=False, identifier=len(jobs))
            jobs.append(j)
        st = [{"t0": time.perf_counter(), "first": None, "last": None, "n": 0, "text": "", "acc": 0, "rej": 0} for _ in jobs]
        gen = state["gen"]
        for j in jobs:
            gen.enqueue(j)
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") != "streaming":
                    continue
                s = st[r["identifier"]]
                now = time.perf_counter()
                tids = r.get("token_ids")
                k = int(tids.numel()) if tids is not None else 0
                if k and s["first"] is None:
                    s["first"] = now
                if k:
                    s["last"] = now
                s["n"] += k
                s["text"] += r.get("text") or ""
                if r.get("eos"):
                    s["n"] = max(s["n"], int(r.get("new_tokens", 0)))
                    s["acc"] = r.get("accepted_draft_tokens", 0); s["rej"] = r.get("rejected_draft_tokens", 0)
                    s["tp"] = r.get("time_prefill"); s["tg"] = r.get("time_generate")
        return st

    def needle_ids(n_tokens):
        r = random.Random(4321)
        cities = ["Lisbon", "Osaka", "Tallinn", "Nairobi", "Quito", "Bergen", "Hanoi", "Tbilisi", "Perth", "Calgary", "Porto", "Krakow"]
        items = ["copper wire", "ceramic tiles", "dried apricots", "bicycle frames", "glass jars", "wool blankets", "steel bolts",
                 "paper reels", "olive oil", "rubber seals", "pine boards", "sea salt"]
        lines, nums = [], []
        while True:
            for _ in range(40):
                nums.append(r.randrange(1000, 9999))
                lines.append(f"Record {len(nums)}: the warehouse in {r.choice(cities)} holds {nums[-1]} crates of {r.choice(items)}.")
            if tok.encode("\n".join(lines))[0].numel() >= n_tokens:
                break
        picks = [3, len(nums) // 2, len(nums) - 2]
        q = ("\n".join(lines) + f"\n\nFrom the records above, state the number of crates in Record {picks[0]}, Record {picks[1]} and "
             f"Record {picks[2]}, in that order, as three numbers separated by commas. Answer with the numbers only.")
        ids = tok.encode(f"[gMASK]<sop><|user|>{q}<|assistant|><think></think>", encode_special_tokens=True)[0].tolist()
        return ids, [nums[p - 1] for p in picks]

    # warm-up
    run([rand_ids(512)], 8, False)

    for n in (args.needle or []):
        ids, want = needle_ids(n)
        for _ in range(args.reps):
            s = run([ids], 40, True)[0]
            got = s["text"].strip()
            ok = all(str(w) in got for w in want)
            rate = round(len(ids) / (s["first"] - s["t0"]), 1)
            print(f"needle {len(ids)} tok: prefill {rate} tok/s, expected {want}, answer {got[:120]!r}, {'OK' if ok else 'MISMATCH'}", flush=True)
            res.setdefault("needle", {}).setdefault(str(len(ids)), []).append({"prefill": rate, "expected": want, "answer": got[:200], "ok": ok})

    for chunk in (args.chunk_sweep or [args.chunk_size]):
        if args.chunk_sweep:
            make_gen(args.num_draft_tokens, chunk)
        for n in args.prefill:
            rates = []
            for _ in range(args.reps):
                s = run([rand_ids(n)], 2, False)[0]
                rates.append(round(n / (s["first"] - s["t0"]), 1))
                print(f"prefill {n} chunk {chunk}: {rates[-1]} tok/s", flush=True)
            key = str(n) if not args.chunk_sweep else f"{n}_chunk{chunk}"
            res["prefill"][key] = {"median": sorted(rates)[len(rates) // 2], "all": rates}
    if args.chunk_sweep:
        make_gen(args.num_draft_tokens, args.chunk_size)

    shown = False
    try:
        from exllamav3.util import gtrace
    except Exception:
        gtrace = None
    import ast, hashlib
    from exllamav3.util import abtest
    variants = []
    for v in (args.ab or [None]):
        if v is None:
            variants.append((None, [])); continue
        parts = v.split("|")
        sets = []
        for kv in parts[1:]:
            path, val = kv.split("=", 1)
            mod, attr = path.split(":")
            sets.append((mod, attr, ast.literal_eval(val)))
        variants.append((parts[0], sets))
    for vname, sets in variants:
        for mod, attr, val in sets:
            if mod == "backend":
                abtest.set_backend([model, draft_model], attr, val)
            else:
                abtest.set_global([model, draft_model], mod, attr, val)
        for ndt in (args.ndt_sweep if args.ndt_sweep is not None else [None]):
            if ndt is not None:
                make_gen(ndt, args.chunk_size)
            if ndt is not None or vname is not None:
                run([chat_ids(999)], 48, True)   # warm the path (graphs instantiate on the third call of a shape)
            tagk = ("" if vname is None else vname + "_") + ("" if ndt is None else f"ndt{ndt}_")
            for ctx in args.dec_ctx:
                for c in args.conc:
                    rounds = []
                    for rep in range(args.reps):
                        ids = [(rand_ids(ctx) if ctx else []) + chat_ids(rep * 10 + i) for i in range(c)]
                        prof = args.nsys and rep == args.reps - 1
                        if prof:
                            import subprocess
                            out = f"{args.nsys_out}.{tagk}ctx{ctx}_C{c}"
                            r = subprocess.run(["nsys", "start", "--session", args.nsys, "--sample=none", "--cpuctxsw=none", "-o", out,
                                                "--force-overwrite", "true"], capture_output=True, text=True)
                            print(f"nsys start rc={r.returncode} {(r.stdout + r.stderr).strip()[:300]}", flush=True)
                        spies = []
                        if args.pyspy and rep == args.reps - 1:
                            import subprocess
                            pids = {"main": os.getpid()}
                            for d, ch in enumerate(getattr(model, "mp_children", []) or []):
                                if ch is not None and getattr(ch, "pid", None):
                                    pids[f"dev{d}"] = ch.pid
                            for name, pid in pids.items():
                                out = f"{args.nsys_out}.pyspy.{tagk}{name}.txt"
                                spies.append(subprocess.Popen(["sudo", "-n", os.path.join(os.path.dirname(sys.executable), "py-spy"), "record",
                                                               "-p", str(pid), "-d", str(int(args.pyspy)), "-r", "500", "-f", "raw", "-o", out],
                                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                            time.sleep(1.0)
                        st = run(ids, None if args.natural else args.gen, True)
                        for sp in spies:
                            sp.wait()
                        if prof:
                            r = subprocess.run(["nsys", "stop", "--session", args.nsys], capture_output=True, text=True)
                            print(f"nsys stop rc={r.returncode} {(r.stdout + r.stderr).strip()[:300]}", flush=True)
                        tot = sum(s["n"] for s in st)
                        span = max(s["last"] for s in st) - min(s["first"] for s in st)
                        per = [(s["n"] - 1) / (s["last"] - s["first"]) for s in st if s["n"] > 1]
                        acc = sum(s["acc"] for s in st); rej = sum(s["rej"] for s in st)
                        sha = hashlib.sha1("\x00".join(s["text"] for s in st).encode()).hexdigest()[:10]
                        rounds.append({"aggregate": round((tot - c) / span, 2), "per_stream": round(sum(per) / len(per), 2), "tokens": tot,
                                       "draft_acc": acc, "draft_rej": rej, "sha": sha})
                        print(f"decode {tagk}ctx={ctx} C{c}: aggregate {rounds[-1]['aggregate']} tok/s, per-stream {rounds[-1]['per_stream']}, "
                              f"{tot} tok, sha {sha}" + (f", draft acc/rej {acc}/{rej}, tok/round {tot / max(1, tot - acc):.2f}" if acc or rej else ""), flush=True)
                        if args.show and not shown:
                            shown = True
                            print("----- sample -----\n" + st[0]["text"][:600].replace("\n", " / ") + "\n------------------", flush=True)
                    agg = sorted(r["aggregate"] for r in rounds)
                    res["decode"][f"{tagk}ctx{ctx}_C{c}"] = {"aggregate_median": agg[len(agg) // 2], "aggregate_max": agg[-1], "rounds": rounds}
        if int(os.environ.get("EXL3_XTIER_SLOTS", "0")) > 0:
            abtest.call_all([model], "exllamav3.util.xtier", "print_summary")
        if gtrace is not None and gtrace.ENABLED and vname is not None:
            abtest.call_all([model], "exllamav3.util.gtrace", "checkpoint", vname)
    if gtrace is not None and gtrace.ENABLED:
        gtrace.flush(); gtrace.dump()
    try:
        abtest.call_all([model], "exllamav3.util.memrep", "print_mem", "end")
    except Exception as e:
        print(f" -- mem report unavailable: {e}", flush=True)
    res["vram_end_mib"] = [round(torch.cuda.mem_get_info(i)[1] / 2**20 - torch.cuda.mem_get_info(i)[0] / 2**20) for i in range(torch.cuda.device_count())]
    print("RESULT " + json.dumps({"tag": args.tag, "prefill": {k: v["median"] for k, v in res["prefill"].items()},
                                  "decode": {k: v["aggregate_median"] for k, v in res["decode"].items()}}), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    try:
        state["gen"].close()
    except Exception:
        pass
    try:
        model.unload()
    except Exception:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
