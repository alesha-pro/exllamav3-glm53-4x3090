#!/bin/bash
# GLM-5.3-Flash on 4x 24 GB GPUs with this exllamav3 fork (tensor parallel). OpenAI API on PORT, model id glm-5.3-flash.
#
#   run_server.sh fast     2.05 bpw, every expert in VRAM, 256K context
#   run_server.sh mix      2.4 bpw mix (expert gate/up at 2 bits, down at 3 bits), 77% of experts in VRAM, 256K
#   run_server.sh quality  3.05 bpw, half of the experts in pinned RAM, 131K
#
# MODELS  directory with the model folders (default: ./models next to this script), or set MODEL to one folder
# VEC     refusal vector (.safetensors) to project out of the residual stream; unset = stock model
# CTX, CQ, AMBS, CHUNK, PORT  context, KV cache bits, state slots, prefill chunk, port
# Extra arguments are passed to serve.py (the last value wins).
H=$(cd "$(dirname "$0")" && pwd)
MODELS=${MODELS:-$H/models}
MODE=${1:-fast}; shift
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
export PATH=$CUDA_HOME/bin:$PATH PYTHONUNBUFFERED=1 TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$H/triton_cache}
export EXL3_HC_FOLD=${EXL3_HC_FOLD:-1}                    # fold the hyper-connection apply / norm launches
export EXL3_TP_SPIN_RECV=${EXL3_TP_SPIN_RECV:-4}          # ranks poll their command pipe between passes
export EXL3_MOE_FUSED_ROWS=${EXL3_MOE_FUSED_ROWS:-32} EXL3_MOE_FUSED_ROWS_WIDE=${EXL3_MOE_FUSED_ROWS_WIDE:-32}   # prefill tiers
export EXL3_VISION_PINNED=${EXL3_VISION_PINNED:-1}        # vision tower weights in RAM
export EXL3_TP_P2P_BIG=${EXL3_TP_P2P_BIG:-1}              # prefill-size reduces and broadcasts GPU to GPU (0: host path)
[ -n "$VEC" ] && export EXL3_CVEC=$VEC
mkdir -p $H/tune
case "$MODE" in
  fast)
    M=${MODEL:-$MODELS/GLM-5.3-Flash-exl3-2.05bpw}
    export EXLLAMAV3_TUNE_CACHE=${EXLLAMAV3_TUNE_CACHE:-$H/tune/tune_fast.bin}
    ARGS=(-m $M -tp -mtp -ndt 1 -cs ${CTX:-262144} -cq ${CQ:-8} -chunk_size ${CHUNK:-4096} -ambs ${AMBS:-4}) ;;
  mix)
    # One merged folder (tools/merge_mix.py). The same model without a merge: MODEL=<2.05 bpw folder>
    # EXL3_MIX_DIR=<3.05 bpw folder> takes the expert down projections from the second quant at load time.
    M=${MODEL:-$MODELS/GLM-5.3-Flash-exl3-2.4bpw-mix}
    export EXLLAMAV3_TUNE_CACHE=${EXLLAMAV3_TUNE_CACHE:-$H/tune/tune_mix.bin}
    export EXL3_XTIER_STATS=${EXL3_XTIER_STATS:-$H/data/stats_own_dec.json}
    export EXL3_XTIER_HOT_FRAC=${EXL3_XTIER_HOT_FRAC:-0.77} EXL3_XTIER_SLOTS=${EXL3_XTIER_SLOTS:-160}
    ARGS=(-m $M -tp -mtp -ndt 1 -cs ${CTX:-262144} -cq ${CQ:-8} -chunk_size ${CHUNK:-4096} -ambs ${AMBS:-2}) ;;
  quality)
    M=${MODEL:-$MODELS/GLM-5.3-Flash-exl3-3.05bpw}
    export EXLLAMAV3_TUNE_CACHE=${EXLLAMAV3_TUNE_CACHE:-$H/tune/tune_quality.bin}
    export EXL3_XTIER_STATS=${EXL3_XTIER_STATS:-$H/data/stats_own_dec.json}
    export EXL3_XTIER_HOT_FRAC=${EXL3_XTIER_HOT_FRAC:-0.50} EXL3_XTIER_SLOTS=${EXL3_XTIER_SLOTS:-480}
    ARGS=(-m $M -tp -mtp -ndt 1 -cs ${CTX:-131072} -cq ${CQ:-8} -chunk_size ${CHUNK:-4096} -ambs ${AMBS:-2}) ;;
  *) echo "mode: fast | mix | quality"; exit 1 ;;
esac
exec ${PYTHON:-python} $H/serve.py "${ARGS[@]}" --vision --max-batch-size 8 --max-rq-tokens ${MAX_RQ:-16384} \
    --host ${HOST:-0.0.0.0} --port ${PORT:-30000} --served-name glm-5.3-flash "$@"
