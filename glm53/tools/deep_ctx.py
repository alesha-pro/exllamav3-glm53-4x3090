# One request near the context limit: a code hidden at the start, the model has to repeat it.
import json, random, sys, time, urllib.request
URL = "http://127.0.0.1:30000"; N_WORDS = int(sys.argv[1]) if len(sys.argv) > 1 else 191000
WORDS = ("time year people way day man thing woman life child world school state family student group country problem "
         "hand part place case week company system program question work government number night point home water").split()
rng = random.Random(7)
lines = [f"Record {i}: " + " ".join(rng.choice(WORDS) for _ in range(10)) + "." for i in range(N_WORDS // 12)]
lines.insert(40, "Record X: the access code is ZEBRA-4417-KIWI.")
prompt = "\n".join(lines) + "\n\nWhat is the access code mentioned in the records above? Answer with the code only."
body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    dt = time.time() - t0
    u = r["usage"]
    print("answer:", repr(r["choices"][0]["message"]["content"]), "| usage:", u, f"| {dt:.0f} s | {u['prompt_tokens']/dt:.0f} tok/s incl. answer")
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read()[:500])
