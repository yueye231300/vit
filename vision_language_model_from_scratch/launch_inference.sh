#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-$SCRIPT_DIR/3b-pt}"
PROMPT="${PROMPT:-what is picture taking about}"
IMAGE_FILE_PATH="${IMAGE_FILE_PATH:-$SCRIPT_DIR/test_images/sea.jpg}"
MAX_TOKENS_TO_GENERATE="${MAX_TOKENS_TO_GENERATE:-1000}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
DO_SAMPLE="${DO_SAMPLE:-False}"
ONLY_CPU="${ONLY_CPU:-False}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: python or python3 was not found in this bash environment." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" "$SCRIPT_DIR/inference.py" \
  --model_path "$MODEL_PATH" \
  --prompt "$PROMPT" \
  --image_file_path "$IMAGE_FILE_PATH" \
  --max_tokens_to_generate "$MAX_TOKENS_TO_GENERATE" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --do_sample "$DO_SAMPLE" \
  --only_cpu "$ONLY_CPU"
