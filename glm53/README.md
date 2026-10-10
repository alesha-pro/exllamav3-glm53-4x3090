# GLM-5.3-Flash on 4x RTX 3090

This branch is exllamav3 1.6.0 (commit `151539c`) plus my changes for one model on one kind of machine: GLM-5.3-Flash (320B parameters, 18B active) on 4 cards with 24 GB each. It serves a 256K context with vision, tool calls and the MTP head at 79 tokens/s in 2 bits, or 43 tokens/s in 3 bits with part of the experts in RAM.

I measured everything below on my rig. Nothing here is tested on other hardware.

## Credits first

Most of what runs here is other people's work.

| Part | Authors | Links |
|---|---|---|
| The engine and the EXL3 format | turboderp | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) |
| The EXL3 quantizations I run | turboderp | [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3) |
| Experts in RAM for this model, the server, the benchmark protocol, the routing statistics | 0xSero | [sybil-solutions/glm53-flash-offload](https://github.com/sybil-solutions/glm53-flash-offload) |
| The idea of 2-bit gate/up with 3-bit down projections | mw00 and PeasantSmith | [Project Maya](https://github.com/mw00/project-maya) |
| The model | Z.ai | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| BF16 teacher logits for the KL numbers | brandonmusic | [GLM-5.3-Flash-BF16-Teacher-Logits](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits) |

0xSero's repo runs this model on 1x 3090 with experts in RAM and on NVMe. I started from it. `serve.py` in this folder is their server with small changes, the cache kernel in `xtier.cu` is based on their `ec_step` / `ec_copy`, and `data/stats_own_dec.json` is their file. Their MIT license is in `LICENSE.glm53-flash-offload`.

## Results

Rig: 4x RTX 3090 at 300 W, PCIe 3.0 x16 on every card, EPYC 7642, 2x 64 GB DDR4-2400 (2 memory channels of 8), NVIDIA open kernel module patched for P2P between the cards.

| Mode | Size | KLD to BF16 | Decode, 1 stream | 2 streams | At 32K depth | Prefill 8K / 32K |
|---|---:|---:|---:|---:|---:|---:|
| `fast`: 2.05 bpw, all experts in VRAM, 256K | 85.1 GB | 0.1214 | 79.1 | 96.3 | 78.2 | 1447 / 1520 |
| `mix`: gate/up 2 bits + down 3 bits, 77% of experts in VRAM, 256K | 98.1 GB | 0.0959 | 65.6 | 70.0 | 54.6 | 1139 to 1141 / 1237 to 1242 |
| `quality`: 3.05 bpw, 50% of experts in VRAM, 131K | 125.2 GB | 0.0473 | 42.7 | 48.6 | 37.5 | 923 / 1051 |
| Stock 1.6.0, 2.05 bpw, tensor parallel, no MTP | | | 66 to 68 | | | 1079 / not measured |
| Stock 1.6.0, 2.05 bpw, layer split | | | 48.3 | | | 931 / not measured |
| Stock 1.6.0, 3.05 bpw, layer split, expert tail on CPU | | | 11.5 | | | 264 / not measured |

Decode and prefill are tokens per second. The 2-stream column is the sum over both streams. `fast` with 4 streams gives 109.8 in total.

How I measured:

- The first 3 rows use the protocol of 0xSero's `bench/sweep.py` through the OpenAI server: answers run to their natural end (1.5K to 4K tokens per stream), 3 runs per prefill point, 2 decode rounds. All 3 modes run with MTP at draft length 1, vision loaded, 8-bit KV cache.
- The stock rows use a short in-process benchmark (`tools/qbench.py`, 384 to 500 tokens).
- KLD is KL(original ‖ quant) in nats over the full vocabulary on the 25-window BF16 panel (51,175 positions), computed through this engine in the mode it serves (`tools/kld_eval.py`). Top-1 agreement with BF16: 88.68%, 89.87% and 93.08%.
- I did not measure task quality.

One request with 233,712 prompt tokens on `mix` returned the code I had hidden at the start of the prompt. It took 185 s, which is 1260 tokens/s of prefill, and VRAM after it was 23,092 to 23,371 MiB of 24,576 per card.

## What I changed in the engine

1. **GPU-to-GPU exchange for prefill** (`exllamav3_ext/parallel/p2p_big.cu`, `EXL3_TP_P2P_BIG=1`). The stock all-reduce goes through pinned host RAM and takes 26 to 29 ms per call on a prefill-size tensor (4096 x 4096). On 2.05 bpw about 70% of a prefill pass was this exchange. Now every rank shares a buffer in its own VRAM over CUDA IPC, and the ranks add their partial sums pairwise in 2 rounds (0↔1 and 2↔3, then 0↔2 and 1↔3). A call takes 7 to 10 ms. Prefill at 32K went from 989 to 1520 tokens/s. The stream hash matches on all 4 ranks after every module, and KLD did not change (0.1215 before, 0.1214 after). It costs 256 MiB of VRAM per card at chunk 4096.
2. **Experts in RAM under tensor parallelism** (`util/xtier.py`, `exllamav3_ext/xtier.cu`). Stock exllamav3 can move experts off the GPU only in layer-split mode. Here every card keeps its most used experts in VRAM and the rest in page-locked RAM that the card reads directly. On a decode step a cold expert is copied into a CLOCK cache in VRAM. One pool per expert size, so mixed quants work.
3. **Mixed quants at load time** (`loader/safetensors.py`). `EXL3_MIX_DIR=<second quant>` replaces the tensors matching `EXL3_MIX_RE` with the tensors of the same name from a second quant. Nothing is written to disk. The default pattern takes the expert down projections. `tools/merge_mix.py` writes the same mix as one folder.
4. **Host overhead in the layer walk.** The temporary tensor cache built a string key on every call, now it is a tuple. The MLA layers read sequence lengths back from the GPU on every pass, now they come from the host. The walk went from 13 ms to about 10 ms per token.
5. **Hyper-connection folding under tensor parallelism** (`EXL3_HC_FOLD=1`). Stock turns it off in TP.
6. **Projection of directions out of the residual stream** (`util/cvec.py`, `EXL3_CVEC=<file>`). This is how the refusal vector is applied. See [abliteration/README.md](abliteration/README.md).
7. **Tracing** (`util/gtrace.py`, `util/memrep.py`, `util/abtest.py`, `util/rtrace.py`): per-module timings on every rank, memory report, A/B switches inside one process, router trace.
8. **GPU-to-GPU all-reduce for decode** (`parallel/p2p.cu`, `EXL3_TP_P2P=1`). It works and it is off, because it is slower than the host path on decode-size tensors: 63.4 against 67.7 tokens/s.

Settings that needed no code:

- `-mtp -ndt 1`. The default draft length is 3, and at 3 the MTP head gave nothing (52.7 to 67.8 tokens/s against 66 to 68 without it). At length 1 it accepts about 1.7 tokens per round: 75.7 on stock, 85.9 with the changes above.
- `EXL3_TP_SPIN_RECV=4`, `EXL3_MOE_FUSED_ROWS=32`, `EXL3_MOE_FUSED_ROWS_WIDE=32`.

Without MTP the fork does not speed up decode (67.7 against 66 to 68). The decode gain shows only with MTP.

## The mix

An expert has 3 matrices: gate, up and down. Project Maya keeps gate and up at 2 bits and gives down more. I tried the same split with turboderp's two quants: everything from 2.05 bpw, expert down projections from 3.05 bpw. The MoE kernels already accepted a separate bit width for down.

| Quant | Size | KLD to BF16 | Top-1 |
|---|---:|---:|---:|
| 2.05 bpw | 85.1 GB | 0.1214 | 88.68% |
| gate/up 2 bits, down 3 bits | 98.1 GB | 0.0959 | 89.87% |
| gate/up 3 bits, down 2 bits | about 109 GB (estimate, not built as a folder) | 0.0779 | 91.26% |
| 3.05 bpw | 125.2 GB | 0.0473 | 93.08% |

The third bit in down gives 34% of the distance from 2.05 to 3.05 for one third of the expert bytes. The third bit in gate and up gives 59% for two thirds. So down is about 17% better per byte, not several times better. I also tried giving the third bit to groups of 7 layers: the gains were between 0.009 and 0.014 per group and they add up almost linearly. In EXL3 quality follows expert bytes closely, and the first and last layers are not special.

The reason to run the mix is speed: only 23% of the experts are in RAM, so there are few cache misses, and it fits 256K.

By size the mix is about 2.37 bpw if you interpolate between turboderp's 2.05 and 3.05 labels.

## Build

```bash
git clone -b glm53-4x3090 https://github.com/alesha-pro/exllamav3
cd exllamav3
glm53/setup_env.sh
```

The script makes a venv with torch 2.8.0+cu128 and builds the extension for `sm_86`. It needs CUDA 12.x with `nvcc` (set `CUDA_HOME`), Python 3.12 and `uv`.

## Models

```bash
mkdir -p glm53/models
hf download turboderp/GLM-5.3-Flash-exl3 --revision 2.05bpw --local-dir glm53/models/GLM-5.3-Flash-exl3-2.05bpw
hf download alesha-pro/GLM-5.3-Flash-abliterated-exl3-2.4bpw-mix --local-dir glm53/models/GLM-5.3-Flash-exl3-2.4bpw-mix
hf download turboderp/GLM-5.3-Flash-exl3 --revision 3.05bpw --local-dir glm53/models/GLM-5.3-Flash-exl3-3.05bpw
```

You need only the folder of the mode you run. My two repos carry the same weights plus the refusal vector: [2.05 bpw](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-exl3-2.05bpw) and [the mix](https://huggingface.co/alesha-pro/GLM-5.3-Flash-abliterated-exl3-2.4bpw-mix).

## Run

```bash
source venv/bin/activate
glm53/run_server.sh fast       # 2.05 bpw, 256K
glm53/run_server.sh mix        # the mix, 256K
glm53/run_server.sh quality    # 3.05 bpw, 131K

VEC=glm53/models/GLM-5.3-Flash-exl3-2.4bpw-mix/glm53-refusal-per-block-3dir.safetensors glm53/run_server.sh mix
```

The server speaks the OpenAI API on port 30000, model id `glm-5.3-flash`. Loading takes 65 to 100 s for `fast`, about 150 s for `mix` and about 230 s for `quality` from NVMe.

Check it with `python glm53/tools/smoke.py http://127.0.0.1:30000`: a chat turn, an image and a tool call.

What the server accepts:

- `chat_template_kwargs: {"enable_thinking": false}` turns thinking off.
- `reasoning_effort` or `chat_template_kwargs.reasoning_effort`: the GLM template knows `low` and `high`. Any other value, and no value at all, means `max`, which thinks 3 to 5 times longer. For agents send `high`.
- Images as OpenAI `image_url` content. Tools in the OpenAI format.
- The prompt cache is reused by prefix. A 25,350-token prompt took 19.8 s cold, 0.2 s when repeated and 0.4 s for the next turn of the same chat.

## Environment variables

| Variable | Default in `run_server.sh` | Meaning |
|---|---|---|
| `EXL3_TP_P2P_BIG` | 1 | Prefill-size reduces and broadcasts go GPU to GPU. Needs peer access between the cards. Set 0 for the stock host path |
| `EXL3_XTIER_STATS` | `data/stats_own_dec.json` | Routing statistics that rank experts as hot or cold. Setting it turns expert tiering on |
| `EXL3_XTIER_HOT_FRAC` | 0.77 for `mix`, 0.50 for `quality` | Share of experts each card keeps in VRAM |
| `EXL3_XTIER_SLOTS` | 160 for `mix`, 480 for `quality` | Cache slots for cold experts on each card |
| `EXL3_MIX_DIR`, `EXL3_MIX_RE` | unset | Second quant folder and the tensor name pattern to take from it |
| `EXL3_CVEC`, `EXL3_CVEC_SCALE` | unset, 1.0 | Vector file to project out, and its strength |
| `EXL3_HC_FOLD` | 1 | Fold the hyper-connection launches |
| `EXL3_TP_SPIN_RECV` | 4 | Ranks poll their command pipe between passes |
| `EXL3_MOE_FUSED_ROWS`, `EXL3_MOE_FUSED_ROWS_WIDE` | 32 | Row thresholds of the fused MoE prefill kernels |
| `EXL3_VISION_PINNED` | 1 | Vision tower weights live in RAM |
| `EXL3_GTRACE` | unset | 1 or 2 turns per-module tracing on |

## Memory

`fast` after loading: 22,975 to 23,762 MiB of 24,576 on every card, and 0.5 to 0.8 GiB free after a 4-stream run at 32K. `mix` at 256K: 22,015 to 22,930 MiB after loading, and the machine had 43 to 52 GB of RAM in use with the server up. `quality`: 22,757 to 23,282 MiB plus about 91 GiB of RAM.

A flat quant above about 2.1 bpw does not fit in 96 GB of VRAM with 256K, MTP and vision. That is why the two larger modes keep experts in RAM.

## Tools

- `tools/smoke.py`: server check.
- `tools/chat_sweep.py`: prefill and decode speed through the chat API on coherent text.
- `tools/kld_eval.py`: KLD against stored BF16 logits, through the engine.
- `tools/merge_mix.py`: write a mixed quant as one folder and compare every tensor with its source.
- `tools/deep_ctx.py`, `tools/prefix_test.py`: one request near the context limit, and prefix cache reuse.
- `tools/qbench.py`: in-process benchmark for A/B work.
- `tools/p2p_test.py`, `tools/p2p_bulk.py`: latency and throughput of the GPU-to-GPU path.
- `abliteration/`: how the refusal vector was found and checked.

## Known limits

- One rig, one model. I ran tensor parallel on exactly 4 cards.
- `EXL3_TP_P2P_BIG=1` needs CUDA peer access between consumer cards. On my rig that is a patched open kernel module. Without it use `EXL3_TP_P2P_BIG=0`, and prefill drops to about 961 / 989 tokens/s at 8K / 32K on `fast`.
- My RAM is 2 channels of DDR4-2400 and my slots are PCIe 3.0. Both make cold experts expensive here: a miss is a 9.4 MB copy over PCIe, about 0.75 ms. A machine with PCIe 4.0 should lose less in `mix` and `quality`. I did not test that.
- `quality` slows down with depth: 37.5 tokens/s at 32K against 42.7 on an empty context. I did not look for the cause.
- A prompt longer than the context returns HTTP 500, not a clear 400.
- With tools in the request the answer text streams, but the tool calls arrive in one piece at the end.
- Expert tiering needs a lot of pinned RAM. At 35% of experts in VRAM the 3.05 bpw quant did not load into my 125 GiB.
- The longest prompt I tested is one 233,712-token request. I did not measure long agent sessions.
- With the refusal vector on, one benchmark point did not finish: a decode after 32K random tokens kept generating for 17 minutes and I stopped it. On coherent text of the same depth answers end normally. I did not find the cause.
