# Bench scripts

All scripts use the Python 3 standard library only. Replace `<host>`, `<port>`, `<box1>`, `<box2>` and `<model>` with your values.

## 1. Build the long-prompt text

```bash
bench/get_context.sh
```

The script downloads the public `README.md` and `CHANGELOG.md` of the [MiaAI-Lab dual-Spark recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) at commit `cd839d0` into `bench/contexte.txt` (105,225 bytes). With this file, "16k" prompts are about 20,300 tokens and "64k" prompts are about 80,600 tokens. With another file, the token counts change.

## 2. Throughput (`banc_matrice.py`)

Start the setup first. Then:

```bash
# full matrix: short, 16k, 64k x 1, 4, 8 requests at once x French prose, code
python3 bench/banc_matrice.py <label> http://<host>:<port>/v1/chat/completions <model> 10

# selected cells, for example 6 requests at once, short and 16k prompts
python3 bench/banc_matrice.py <label> http://<host>:<port>/v1/chat/completions <model> 10 --flux 6 --longueurs court,16k

# English prose, short prompts
python3 bench/banc_matrice.py <label> http://<host>:<port>/v1/chat/completions <model> 10 --prose-en --flux 1,4,6,8 --longueurs court

# one model per box: perfect client-side alternation between the two boxes
python3 bench/banc_matrice.py <label> http://<box1>:<port>/v1/chat/completions <model> 10 \
  --urls http://<box1>:<port>/v1/chat/completions,http://<box2>:<port>/v1/chat/completions \
  --flux 1,4,6,8,12 --longueurs court,16k
```

Each request sends `"priority": 100` (vLLM: a high number = low priority) and `chat_template_kwargs.enable_thinking: false`. Results go to `bench/resultats.jsonl`, one line per cell (see `data/results.csv` for the column meanings).

Run only one bench job for each box at a time. Two jobs on the same box give wrong numbers.

## 3. Very long prompts (`needle.py`)

```bash
python3 bench/needle.py http://<host>:<port>/v1/chat/completions <model>
```

It needs an engine with a context limit above 800,000 tokens. We used the dual-Spark vLLM recipe with `MAX_MODEL_LEN=1048576` and `YARN_ENABLE=true` (YaRN x4).

## 4. Decisions (`decisions/`)

1. Clone [JevBench](https://github.com/fstandhartinger/jevbench) into `bench/decisions/jevbench`.
2. Start the adapter in front of your engine:
   `QWEN_URL=http://<host>:<port> QWEN_MODEL=<model> python3 bench/decisions/qwen_systemone.py 8021`
   For TensorFold, add `QWEN_PRIORITY=background`. For the "long text first" test, add `QWEN_CONTEXTE=bench/contexte.txt`.
3. Run the 3 public files: `bench/decisions/run_jevbench.sh <label> http://127.0.0.1:8021 <label>`
4. Score: `python3 bench/decisions/score.py <label>`

The "long text first" runs use the hard file only. Run it with the JevBench CLI and `--tasks datasets/public/hard.jsonl`, as in `run_jevbench.sh`.

## Engine settings we used

- **vLLM, one box**: MiaAI-Lab single-Spark recipe `b843911`, `MAX_NUM_SEQS=6`, MTP with 3 draft tokens. We added small local changes: network settings, two model names, a fix in the health probe, and a memory-guard setting. Our MTP draft vocabulary comes from our own outputs.
- **vLLM, across both boxes**: MiaAI-Lab dual-Spark recipe `cd839d0`, recipe defaults (`MAX_NUM_SEQS=8`, MTP with 3 draft tokens, English-and-code draft vocabulary). We changed only the network settings, the model and its served name.
- **TensorFold, one box** (0.6.5): `tensorfold serve Mia-AiLab/Qwen3.8-Flash-Next-NVFP4 --parallel 5 --context 262144 --kv-dtype int8 --mtp-drafts 6 --mtp-confidence 0.60 --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 32768 --thinking`, with `TENSORFOLD_PREFILL_ROWS=2048`, `TENSORFOLD_MTP_COPY=1`, `TENSORFOLD_MEMORY_RESERVE_GIB=2`.
- **TensorFold, across both boxes** (0.6.5): `tensorfold serve Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP --tp 2 --master <box1 cable IP> --rank <0|1> --parallel 8 --context 262144 --kv-dtype int8 --mtp-drafts 6 --mtp-confidence 0.60 --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 32768 --ple-on-ssd --thinking`, same environment, plus NCCL over the two ConnectX-7 ports (`NCCL_IB_MERGE_NICS=1`, 4 channels).
