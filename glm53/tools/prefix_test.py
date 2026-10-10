# Prefix reuse check: same long prompt twice, then the same prefix with a new tail, then a second turn.
import json, random, time, urllib.request
URL = "http://127.0.0.1:30000"
W = "time year people way day man thing woman life child world school state family student group country problem".split()
rng = random.Random(int(time.time()))
doc = "\n".join(f"Record {i}: " + " ".join(rng.choice(W) for _ in range(10)) + "." for i in range(1600))
def ask(messages, tag):
    body = {"model": "glm-5.3-flash", "messages": messages, "temperature": 0, "max_tokens": 24,
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900).read())
    print(f"{tag}: {time.time()-t0:.1f} s, prompt {r['usage']['prompt_tokens']} tokens, usage {r['usage']}")
    return r["choices"][0]["message"]["content"]
m1 = [{"role": "user", "content": doc + "\n\nHow many records are there? Number only."}]
a = ask(m1, "1 cold")
ask(m1, "2 same prompt again")
ask([{"role": "user", "content": doc + "\n\nWhat is the first word of record 3? One word."}], "3 same document, new question")
ask(m1 + [{"role": "assistant", "content": a}, {"role": "user", "content": "And what is the last record number?"}], "4 second turn of chat 1")
