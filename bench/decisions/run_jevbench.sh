#!/bin/bash
# Run the 3 public JevBench files (easy 48, original 72, hard 111 = 231 decisions) against one endpoint.
#   run_jevbench.sh <label> <endpoint> <model> [key_env]
# Needs a clone of https://github.com/fstandhartinger/jevbench in ./jevbench (next to this script).
# <endpoint> is the qwen_systemone.py server, for example http://127.0.0.1:8021
set -u
cd "$(dirname "$0")/jevbench"
NAME=$1; EP=$2; MODEL=$3; KEY=${4:-}
OUT=../runs/$NAME; mkdir -p "$OUT"
for f in easy original hard; do
  python3 -m jevbench.cli run --tasks datasets/public/$f.jsonl --adapter typesafe \
    --endpoint "$EP" --model "$MODEL" --key-env "$KEY" \
    --results "$OUT/$f.jsonl" --ledger "$OUT/ledger.jsonl" --raw-dir "$OUT/raw" \
    --reserve-usd 0 --cap-usd 1 2>&1 | tail -1
done
echo "DONE $NAME"
