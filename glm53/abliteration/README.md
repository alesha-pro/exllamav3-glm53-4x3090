# GLM-5.3-Flash: refusals removed by projection at run time

The weights stay as published. After every block the engine subtracts the components of the residual stream along 3 directions found for that block:

```text
x = x - scale * (x Vᵀ) V
```

`V` holds the 3 orthonormal directions of the block, and the same operation runs on all 4 hyper-connection streams. Without `EXL3_CVEC` the engine runs the stock model.

The vector file is `glm53-refusal-per-block-3dir.safetensors` (2.2 MB, blocks 1 to 44, SHA256 `f07b2cce3f91a36658c9706301088ba9c0140f3dc032cc26937136c1779f1764`). It is in my Hugging Face repos: [2.05 bpw](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-exl3-2.05bpw) and [the mix](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-exl3-2.4bpw-mix). A version for Project Maya is in the [Maya-M repo](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-Maya-M-GGUF).

```bash
VEC=/path/to/glm53-refusal-per-block-3dir.safetensors glm53/run_server.sh fast
```

`EXL3_CVEC_SCALE` sets the strength. 1.0 removes the component.

## Results

The prompt sets are 64 held-out AdvBench requests and the 82 requests of JailbreakBench, which took no part in finding the directions. Decoding is greedy and everything goes through the API of the running server. The judge is the stock model without the vector, which reads the request and the answer and labels the answer (`judge_ls.py`).

`fast` mode (2.05 bpw, 256K, MTP, vision, 4 streams), all 146 requests in each of 3 modes, 438 answers:

| Mode | Refused without the vector | Refused with the vector | Complied | Disputes the premise | Cut off while thinking |
|---|---:|---:|---:|---:|---:|
| Thinking off | 144 of 146 | 0 | 143 | 3 | 0 |
| Thinking, effort `high`, up to 6000 tokens | 32 of 32 | 0 | 142 | 2 | 2 |
| Thinking, effort `max` (the template default), up to 12,000 tokens | 46 of 48, 2 unfinished | 0 | 135 | 2 | 9 |

The column "without the vector" is the same quant in layer-split mode, and for the two thinking modes I ran only the first 32 and the first 48 requests there.

An answer that disputes the premise is one where the model takes the request and says its premise is false. Asked for proof that the Earth is flat, it answers that there is none and lists what flat-Earth believers usually argue. 5 of the 7 are requests with a false premise and 2 are "build a game / an app" tasks. A strict 2-class judge (refused or complied) put 1 of these 7 into refusals and 6 into complied, so by that judge the result is 1 of 438.

10 of the 11 answers cut off while thinking are "create / develop a site, an app, a game, a bot, a tool" tasks. The model writes 3,000 to 7,000 words of reasoning and does not close the thinking block within the token budget. The text has no loops, and the judge read the reasoning and put all 11 into "working on the task".

A plain regex counter for refusal phrases fired 2 times in 438 and both were false. In one the phrase was inside a letter the model wrote on request, and the other was "I can't connect to systems myself, but here is how it is done".

The other two engines with the same directions, 178 answers each (146 without thinking, 32 with thinking):

| Setup | Refused | Disputes the premise | Cut off while thinking |
|---|---:|---:|---:|
| This fork, the mix (gate/up 2 bits, down 3 bits), 256K | 0 | 4 | 5, all "working on the task" |
| Project Maya, Maya-M, 256K | 2 | 4 | 0 |

The 2 refusals on Maya are requests about suicide and self-harm. The model turns them into text about prevention. I left these 2 as they are.

The directions were found on the 2.05 bpw quant, and that is where the result is cleanest.

## Finding the directions

1. **Activations** (`resid_dump.py`). 256 harmful and 256 harmless requests in 5 framings: thinking off, thinking on, a short system prompt, and a coding-agent system prompt in both modes. At the last prompt token I record the stream after each of the 45 blocks: 4 hyper-connection streams of 4096. Quant 2.05 bpw, layer-split mode.
2. **Direction** (`directions.py`). Per block: mean of harmful minus mean of harmless, averaged over streams and framings. Then I remove the components along the mean harmless states. Without that step the projection also moves ordinary prompts. From block 17 on the direction separates held-out harmful and harmless requests with no errors (AUROC 1.000).
3. **One direction is not enough with thinking on.** The "thinking off" and "thinking on" directions agree only at cosine 0.4 to 0.5. With 1 direction per block the model still refused 8 of 16 requests when it was thinking.
4. **The third direction is the end of the model's own thinking** (`think_capture.py`). The stock model thinks about 96 harmful and 96 harmless requests, and I record the stream at the closing `</think>` token. At that point the decision is already made. 92 harmful and 40 harmless runs closed their thinking within 900 tokens. This direction differs from the first two (cosine 0.2 to 0.6) and removes the refusals that were left.
5. **The final file** has 3 orthonormal directions per block (thinking off, thinking on, end of thinking) for blocks 1 to 44.

Selection runs, 2.05 bpw, layer-split, judged:

| Vector | Thinking off, refused of 64 + 82 | Thinking on, refused of the first 16 AdvBench |
|---|---|---|
| None | 64 + 80 | 15 |
| 1 direction per block | 1 + 3 | 8, and 3 unfinished |
| The block 27 direction on every block | 2 + 2 | not measured |
| 2 directions per block | 0 + 0 | 3, and 4 unfinished |
| 2 directions, scale 1.5 | 1 + 1 | 1, and 4 unfinished |
| 2 directions, scale 2.0 | 0 + 0 | 1, and 5 unfinished |
| 5 directions (one per framing) | 0 + 1 | not measured |
| **3 directions, with the end of thinking** | 0 + 0 through the API | **0, and 6 unfinished** |

A scale above 1.0 does not remove the refusals in thinking mode. The missing direction had to be found.

The GLM chat template has 3 thinking levels: `low`, `high` and `max`. With no level set it uses `max`, and answers come out 3 to 5 times longer.

## Cost

KLD to BF16 on the 25-window panel, 2.05 bpw, layer-split:

| Vector | KLD | Top-1 |
|---|---:|---:|
| None | 0.1219 | 88.68% |
| 1 direction per block | 0.1251 | |
| 2 directions | 0.1274 | |
| 3 directions (final) | 0.1318 | 88.32% |

On harmless requests the number of refusals did not grow, and no run had empty or looping answers. The server with the vector passes `tools/smoke.py`: chat, tool call, image.

Speed in `fast` mode, same protocol as the main table: decode 74.95 tokens/s on 1 stream (79.07 without the vector), 93.2 on 2 (96.3), 113.1 on 4 (109.8). Prefill 1389 at 8K (1447) and 1467 at 32K (1520). That is 3 to 5% slower on 1 stream.

The mix and Maya-M at 256K through `tools/chat_sweep.py`:

| | Without the vector | With the vector |
|---|---|---|
| This fork, the mix: decode 1 stream / 2 streams / at 32K | 53.7 / 59.7 / 59.7 | 52.4 / 62.7 / 58.5 |
| This fork, the mix: prefill 8K / 32K | 1066 to 1236 / 1291 to 1292 | 1180 to 1193 / 1250 to 1254 |
| Maya-M: decode 1 stream / at 32K | 58.4 / 57.2 | 56.1 / 55.9 |
| Maya-M: prefill 8K / 32K | 1078 to 1113 / 1860 to 1896 | 1119 to 1153 / 1938 to 1953 |

## Engine side

`exllamav3/util/cvec.py`:

- after every block the stream `x[.., 4, 4096]` is projected with the directions of that block;
- on decode a block defers its last stream update to the next block (`EXL3_HC_FOLD`). The hook writes that update first. Without this the projection read a stale stream and the last block reached the head unprojected;
- recording for measurements (`capture_begin` / `capture_end`) and switching the vector inside one process (`configure`).

## Scripts

| Script | What it does |
|---|---|
| `resid_dump.py` | Records activations at the last prompt token |
| `think_capture.py` | Records activations at the end of the stock model's own thinking |
| `directions.py` | Analyzes the recordings and writes vector files (`analyze`, `write`, `single`, `multi`) |
| `sweep_ls.py` | Counts refusals for several vectors in one model load |
| `eval_refusal.py` | Runs a prompt set through the API of a live server, with thinking effort and parallel requests |
| `judge_ls.py` | Uses the stock model as the judge. It has 2 classes by default and `--three` adds "disputes the premise". An answer cut off while thinking is judged on its reasoning |

The prompt sets are not in this repository. The scripts read them as JSON lists of strings. The answers to harmful prompts are not published.

## Limits

- The judge is this same model without the vector, with 2 labeling schemes. There are no human labels.
- 146 prompts from 2 public sets. Other kinds of requests may behave differently.
- The speed point "decode after 32K random tokens" did not finish with the vector on. See the main README.
