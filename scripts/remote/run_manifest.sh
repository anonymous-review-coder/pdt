#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/exp_outputs/r-2026-pdt}"

usage() {
  cat >&2 <<EOF
usage: run_manifest.sh <manifest> [run_id] [--gpu ID] [--dry-run] [--skip-predictions]

Environment:
  OUTPUT_ROOT      output root, default: ${OUTPUT_ROOT}
  GPU              default GPU id if --gpu is omitted; if unset, keep manifest env
EOF
}

MANIFEST=""
RUN_ID=""
GPU_ID="${GPU:-}"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      EXTRA_ARGS+=("$1")
      shift
      ;;
    --gpu)
      if [[ -z "${2:-}" ]]; then
        echo "--gpu requires a GPU id." >&2
        exit 1
      fi
      GPU_ID="$2"
      shift 2
      ;;
    --skip-predictions)
      EXTRA_ARGS+=(--set args.skip_predictions=true)
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -z "$MANIFEST" ]]; then
        MANIFEST="$1"
      elif [[ -z "$RUN_ID" ]]; then
        RUN_ID="$1"
      else
        echo "Unexpected positional argument: $1" >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$MANIFEST" ]]; then
  usage
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python command not found; activate the intended environment before running this script." >&2
  exit 1
fi
PYTHON_CMD=(python -u)

cd "$ROOT"
CMD=("${PYTHON_CMD[@]}" -m protocol.runners.run_manifest --manifest "$MANIFEST" --output-root "$OUTPUT_ROOT")
if [[ -n "$RUN_ID" ]]; then
  CMD+=(--run-id "$RUN_ID")
fi
if [[ -n "$GPU_ID" ]]; then
  CMD+=(--set "env.CUDA_VISIBLE_DEVICES=$GPU_ID")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
