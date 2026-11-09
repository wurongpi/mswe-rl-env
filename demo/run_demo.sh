#!/usr/bin/env bash
# Offline end-to-end demo: scripted model -> recording proxy -> mini-swe-agent -> sandbox -> tests -> samples.
# No GPU, no model weights. Use BACKEND=docker to run the sandbox in containers (OrbStack / Colima).
#
#   bash demo/run_demo.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}
BACKEND=${BACKEND:-local}
OUT=runs/demo

mkdir -p "$OUT"
$PY -m mswe_rl.fake_server --script demo/scripts.yaml --port 8000 &
FAKE=$!
$PY -m mswe_rl.proxy --upstream http://127.0.0.1:8000 --port 8100 --record-dir "$OUT/calls" &
PROXY=$!
trap 'kill $FAKE $PROXY 2>/dev/null || true' EXIT
for _ in $(seq 50); do curl -sf http://127.0.0.1:8100/health >/dev/null && break; sleep 0.2; done

echo "== 1. mini-swe-agent rollouts (3 tasks x group 2) =="
$PY -m mswe_rl.rollout --agent mini --model scripted --backend "$BACKEND" \
    --group-size 2 --concurrency 3 --out "$OUT/samples.jsonl" --color

echo
echo "== 2. same task with history truncation (keep last 2 messages) =="
$PY -m mswe_rl.rollout --agent simple --max-history 2 --only fix_fizzbuzz --model scripted \
    --backend "$BACKEND" --group-size 1 --out "$OUT/truncated.jsonl" --color

echo
echo "== 3. what the trainer sees (green = trained tokens, gray = masked) =="
$PY -m mswe_rl.inspect_samples "$OUT/samples.jsonl" --tokenizer toy --max-chars 160 --color
