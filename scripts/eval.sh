#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ $# -eq 0 || "$1" == "--help" ]]; then
  echo 'Usage: bash scripts/eval.sh --model /path/to/hf-model --out /path/to/new-result [--step 150]'
  exit 0
fi
REQUESTED_EVAL_GPU=${EVAL_GPU:-}
if [[ -f .env ]]; then set -a; source .env; set +a; fi
# A paired launch's assigned device must take precedence over a standalone
# EVAL_GPU default in .env, or one arm could occupy the other arm's GPU.
export CUDA_VISIBLE_DEVICES="${REQUESTED_EVAL_GPU:-${EVAL_GPU:-${CUDA_VISIBLE_DEVICES:-}}}"
: "${CUDA_VISIBLE_DEVICES:?Explicitly set your allocated GPU}"
export RAY_ADDRESS=local
: "${ALFWORLD_DATA:?Set the ALFWorld asset directory}"
exec "${OPSD_PYTHON:-python}" src/evaluation/run_seen140.py --assets "$ALFWORLD_DATA" "$@"

