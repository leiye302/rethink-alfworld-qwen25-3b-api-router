#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ $# -eq 0 || "$1" == --help ]]; then
  echo 'Usage: bash scripts/reproduce.sh --check | --run --allow-api-cost [--resume]'
  echo 'Runs ONE declared ablation, then exports and evaluates fixed seen140.'
  echo 'No additional baseline is launched. External API variant skips SFT.'
  exit 0
fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
if [[ "$1" == --check && $# -eq 1 ]]; then
  exec "${SFT_PYTHON:-python}" src/tools/check.py
fi
if [[ "$1" != --run || "${2:-}" != --allow-api-cost ]]; then
  echo 'GPU training/API calls require --run --allow-api-cost.' >&2; exit 2
fi
: "${ALFWORLD_DATA:?Set the ALFWorld asset directory}"
: "${RETHINK_API_KEY:?Provide your own API key}"
: "${RETHINK_API_ENDPOINT:?Provide your own HTTPS endpoint}"
"${SFT_PYTHON:-python}" src/tools/check.py --assets
shift
exec "${SFT_PYTHON:-python}" src/tools/run_pair.py "$@"
