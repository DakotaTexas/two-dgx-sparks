#!/bin/bash
# Build the long-prompt context file used by banc_matrice.py and needle.py.
# It is the public README.md + CHANGELOG.md of the MiaAI-Lab dual-Spark recipe at commit cd839d0.
# We do not copy that text into this repository: this script downloads it.
set -euo pipefail
cd "$(dirname "$0")"
BASE=https://raw.githubusercontent.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/cd839d0
curl -fsSL "$BASE/README.md" > contexte.txt
curl -fsSL "$BASE/CHANGELOG.md" >> contexte.txt
echo "contexte.txt: $(wc -c < contexte.txt) bytes (expected 105225)"
echo "sha256: $(shasum -a 256 contexte.txt | cut -d' ' -f1)"
echo "expected: 70c678c44ed977d15a6dd7d4a0f2cf3dea079ac9c6db5f580fbe92d48a1c1e1c"
