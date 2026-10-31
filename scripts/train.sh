#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ $# -eq 0 || "$1" == --help ]]; then
  echo 'Usage: bash scripts/train.sh {sft|opsd} [options]'
  echo 'One allocated 80/96GB GPU. OPSD requires --allow-api-cost.'
  echo 'External-router variant deliberately skips/disallows SFT.'
  exit 0
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
STAGE=$1; shift
PY=${OPSD_PYTHON:-python}
if [[ "$STAGE" == sft ]]; then PY=${SFT_PYTHON:-python}; fi
CUDA_VISIBLE_DEVICES=$("$PY" src/gpu_layout.py --stage "$STAGE")
export CUDA_VISIBLE_DEVICES RAY_ADDRESS=local TOKENIZERS_PARALLELISM=false
VARIANT=$("$PY" -c "import sys;sys.path.insert(0,'src');from ablation import contract;print(contract()['variant'])")
if [[ "$STAGE" == sft ]]; then
  if [[ "$VARIANT" != trigger_only ]]; then
    echo 'External API ablation starts from the base and must not run Route-SFT.' >&2; exit 2
  fi
  unset RETHINK_API_KEY
  export PYTHONPATH="$ROOT/src/sft" HF_HUB_OFFLINE=1
  "$PY" -m route_bce.preflight --data "$ROOT/data/route_sft" \
    --tokenizer "$ROOT/checkpoints/Qwen2.5-3B-Instruct" \
    --supervision-mode trigger_only --output "$ROOT/reports/cpu_token_preflight.json"
  export ALLOW_ROUTE_SEMANTIC_BCE_TRAINING=1
  exec "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
    --max-restarts=0 --module route_bce.train --root "$ROOT" --config "$ROOT/configs/sft.json" "$@"
elif [[ "$STAGE" == opsd ]]; then
  : "${ALFWORLD_DATA:?Set the ALFWorld asset directory}"
  if [[ ! -f .deps/sdar/.rethink_setup.json ]]; then
    echo 'Run python src/tools/setup.py --framework first.' >&2; exit 2
  fi
  RUN="$ROOT/checkpoints/opsd"
  MODEL="$ROOT/checkpoints/route_sft/global_step_2000"
  if [[ "$VARIANT" == external_api_router ]]; then MODEL="$ROOT/checkpoints/Qwen2.5-3B-Instruct"; fi
  if [[ ! -f "$RUN/setup.json" ]]; then
    "$PY" src/opsd/prepare.py --assets "$ALFWORLD_DATA" --model "$MODEL" --run-dir "$RUN"
  fi
  exec "$PY" src/opsd/run.py --run-dir "$RUN" "$@"
else
  echo 'Unknown stage; expected sft or opsd.' >&2; exit 2
fi
