#!/usr/bin/env python3
"""Speed protocol over the plain OpenAI chat API, so different engines can be measured the same way.

Mirrors glm53-flash-offload's bench/sweep.py (same topics, answers run to their natural end, thinking off,
temperature 0) but needs only POST /v1/chat/completions with streaming:
  prefill N   a natural-text prompt of about N tokens, rate = prompt_tokens / time to first token
  decode Cn   n requests at once, aggregate = tokens / span, per-stream = (tokens - 1) / (last - first)
  decode @D   one request behind a D-token prefix
Every prompt starts with a fresh nonce, so no prefix cache can answer it.

usage: chat_sweep.py --url http://127.0.0.1:30000 --out result.json [--prefill 8192 32768] [--conc 1 2 4] [--depth 32768]
"""
import argparse, json, random, secrets, statistics, threading, time, urllib.request

TOPICS = ["how a four-stroke engine works", "the causes of the French Revolution", "how TCP congestion control works",
          "the life cycle of a star", "how vaccines train the immune system", "the history of the printing press",
          "how public-key cryptography works", "the geology of volcanoes", "the rules and strategy of chess openings",
          "how GPS determines position", "the history of the bicycle", "how noise-cancelling headphones work",
          "the water cycle", "how databases implement transactions"]
WORDS = ("time year people way day man thing woman life child world school state family student group country problem "
         "hand part place case week company system program question work government number night point home water room "
         "mother area money story fact month lot right study book eye job word business issue side kind head house "
         "service friend father power hour game line end member law car city community name president team minute idea "
         "body information back parent face others level office door health person art war history party result change "
         "morning reason research girl guy moment air teacher force education").split()
NONCE = secrets.token_hex(8)


def stream(url, model, messages, max_tokens, timeout=7200):
    body = {"model": model, "messages": messages, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); times = []; usage = {}; timings = {}; finish = None; chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            usage = ev.get("usage") or usage
            timings = ev.get("timings") or timings
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                text = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if text:
                    times.append(time.perf_counter()); chars += len(text)
                finish = ch.get("finish_reason") or finish
    return {"t0": t0, "first": times[0] if times else None, "last": times[-1] if times else None, "chunks": len(times),
            "prompt_tokens": usage.get("prompt_tokens") or timings.get("prompt_n"),
            "completion_tokens": usage.get("completion_tokens") or timings.get("predicted_n") or len(times),
            "finish": finish, "chars": chars, "timings": timings or None}


def filler(n_words, seed):
    rng = random.Random(seed)
    lines = []
    for i in range(n_words // 12):
        lines.append(f"Record {i}: " + " ".join(rng.choice(WORDS) for _ in range(10)) + ".")
    return "\n".join(lines)


def question(i):
    return f"Write a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} (variant {i})."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 32768])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--depth", type=int, nargs="*", default=[32768])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()
    res = {"url": a.url, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "nonce": NONCE, "prefill": {}, "decode": {}}
    save = lambda: json.dump(res, open(a.out, "w"), indent=1)
    seq = [0]

    def nonce():
        seq[0] += 1
        return f"[cache-bust nonce: {NONCE}-{seq[0]}]\n"

    # tokens per filler word, measured on this server's tokenizer
    probe = stream(a.url, a.model, [{"role": "user", "content": nonce() + filler(1200, 1) + "\n\nReply with OK."}], 4)
    per_word = probe["prompt_tokens"] / 1200
    res["tokens_per_word"] = round(per_word, 3)
    print(f"probe: {probe['prompt_tokens']} prompt tokens for 1200 words", flush=True)

    for n in a.prefill:
        rows = []
        for rep in range(a.reps):
            text = nonce() + filler(int(n / per_word), 1000 * n + rep) + "\n\nReply with the single word OK."
            r = stream(a.url, a.model, [{"role": "user", "content": text}], 4)
            rate = r["prompt_tokens"] / (r["first"] - r["t0"])
            rows.append({"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["first"] - r["t0"], 3), "tok_s": round(rate, 1)})
            print(f"prefill {n}: {rate:.1f} tok/s ({r['prompt_tokens']} tokens, ttft {r['first'] - r['t0']:.2f} s)", flush=True)
        res["prefill"][str(n)] = {"tok_s": statistics.median(x["tok_s"] for x in rows), "runs": rows}; save()

    def run_conc(c, rnd, prefix=""):
        out = [None] * c

        def one(i):
            out[i] = stream(a.url, a.model, [{"role": "user", "content": nonce() + prefix + question(rnd * 100 + i)}], a.max_tokens)
        ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
        [t.start() for t in ths]; [t.join() for t in ths]
        tot = sum(o["completion_tokens"] for o in out)
        span = max(o["last"] for o in out) - min(o["first"] for o in out)
        per = [(o["completion_tokens"] - 1) / (o["last"] - o["first"]) for o in out if o["completion_tokens"] > 1]
        return {"aggregate": round(tot / span, 2), "per_stream_mean": round(statistics.mean(per), 2), "tokens": tot,
                "finish": sorted({str(o["finish"]) for o in out}), "ttft_max": round(max(o["first"] - o["t0"] for o in out), 2)}

    for c in a.conc:
        rounds = [run_conc(c, r) for r in range(a.reps)]
        res["decode"][f"C{c}"] = {"aggregate": statistics.median(r["aggregate"] for r in rounds),
                                   "per_stream_mean": statistics.median(r["per_stream_mean"] for r in rounds), "rounds": rounds}
        print(f"decode C{c}: aggregate {res['decode'][f'C{c}']['aggregate']:.2f} tok/s, per-stream "
              f"{res['decode'][f'C{c}']['per_stream_mean']:.2f}, tokens {[r['tokens'] for r in rounds]}", flush=True); save()

    for d in a.depth:
        pre = filler(int(d / per_word), 77 + d) + "\n\nThe records above are filler. Ignore them.\n\n"
        rounds = [run_conc(1, 90 + i, prefix=pre) for i in range(a.reps)]
        res["decode"][f"C1@{d}"] = {"aggregate": statistics.median(r["per_stream_mean"] for r in rounds), "rounds": rounds}
        print(f"decode C1@{d}: {res['decode'][f'C1@{d}']['aggregate']:.2f} tok/s, tokens {[r['tokens'] for r in rounds]}", flush=True); save()

    res["status"] = "DONE"; res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z"); save()
    print("SWEEP DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
