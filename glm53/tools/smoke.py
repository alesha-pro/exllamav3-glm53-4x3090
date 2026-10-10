#!/usr/bin/env python3
"""Server smoke test: chat, thinking off, vision (generated image), tool call. smoke.py [url]"""
import base64, io, json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30000"

def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r, time.time() - t

r, dt = post("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
                                      "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
print("chat:", repr(r["choices"][0]["message"]["content"][:80]), r["usage"], f"{dt:.1f}s")

from PIL import Image, ImageDraw
im = Image.new("RGB", (448, 448), (20, 60, 200))
d = ImageDraw.Draw(im)
d.rectangle([60, 60, 220, 220], fill=(230, 30, 30))
d.ellipse([250, 250, 400, 400], fill=(250, 220, 40))
buf = io.BytesIO(); im.save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
r, dt = post("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": url}},
    {"type": "text", "text": "Describe the shapes and their colors in this image in one sentence."}]}],
    "max_tokens": 300, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
print("vision:", repr(r["choices"][0]["message"]["content"][:300]), r["usage"], f"{dt:.1f}s")

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r, dt = post("/v1/chat/completions", {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "What's the weather in Paris right now?"}],
                                      "tools": tools, "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
print("tool:", r["choices"][0]["message"].get("tool_calls"), r["choices"][0]["finish_reason"], f"{dt:.1f}s")
