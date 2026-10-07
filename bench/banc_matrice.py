#!/usr/bin/env python3
"""Throughput bench used for "Two DGX Sparks: which setup?". Same bench for every setup.

    python3 banc_matrice.py <label> <chat_completions_url> <model> [reps] [options]

Options:
    --court-seul               short prompts only
    --flux 1,4,6,8,12          requests at once (default: 1, 4, 8)
    --longueurs court,16k,64k  prompt lengths (default with --flux: court)
    --prose-en                 English prose task instead of French prose + code
    --urls URL1,URL2           send request k to URL[k % n] (perfect client-side alternation;
                               used for "one model per box")

Matrix: prompt length {court (short), 16k, 64k} x requests at once x task {prose FR, code EN}.
Reasoning OFF, max 450 output tokens, temperature 0.7. Output tokens come from the server
`usage` field (speculative decoding sends several tokens per chunk, so chunk counts are wrong).
Long prompts start with a random salt, so the prefix cache never helps (cold prefill, worst case).
"16k" is really about 20,300 prompt tokens and "64k" about 80,600 with the context file below.

Context file: BENCH_CONTEXT (default: contexte.txt next to this script). Build it with
get_context.sh. Output: one JSON line per cell in BENCH_RESULTS (default: resultats.jsonl
next to this script) and one readable line on stdout.
Optional: BENCH_KEY is sent as a Bearer token (for a gateway that needs a key).
"""
import json, os, random, statistics as st, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL, URL, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
URLS = sys.argv[sys.argv.index("--urls") + 1].split(",") if "--urls" in sys.argv else []
REPS = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 5
RAW = open(os.environ.get("BENCH_CONTEXT", os.path.join(HERE, "contexte.txt"))).read()
RESULTS = os.environ.get("BENCH_RESULTS", os.path.join(HERE, "resultats.jsonl"))
CTX = {"16k": RAW[:60000], "64k": (RAW * 3)[:240000]}
# The prompt texts are kept exactly as run, so the numbers stay comparable.
TASKS = {
    "prose FR": "Rédige en français un paragraphe argumenté d'environ 300 mots sur l'intérêt, pour une PME "
                "européenne de cybersécurité, d'héberger ses propres modèles de langue. Variante %d.",
    "code EN": "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety, "
               "type hints and docstrings, followed by pytest unit tests. Variant %d.",
}
TASK_EN = "Write an argued paragraph of about 300 words on why a European cybersecurity SME should host its own language models. Variant %d."
TASK_EN_LONG = ("Using the documentation above, write an argued paragraph of about 300 words explaining to an executive "
                "when to use two DGX Sparks instead of one. Variant %d.")
TASKS_LONG = {
    "prose FR": "En t'appuyant sur la documentation ci-dessus, rédige en français un paragraphe argumenté "
                "d'environ 300 mots expliquant à un dirigeant quand utiliser deux DGX Spark plutôt qu'un. Variante %d.",
    "code EN": "Using the documentation above, write a complete Python module that parses the .env settings it "
               "describes, validates them with type hints and docstrings, followed by pytest unit tests. Variant %d.",
}


def one(prompt, url=None):
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 450,
            "temperature": 0.7, "priority": 100, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    headers = {"content-type": "application/json"}
    if os.environ.get("BENCH_KEY"):
        headers["authorization"] = "Bearer " + os.environ["BENCH_KEY"]
    req = urllib.request.Request(url or URL, data=json.dumps(body).encode(), headers=headers)
    t0 = time.time(); t1 = None; toks = 0; ptoks = 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and t1 is None:
                t1 = time.time()
            if d.get("usage"):
                toks = d["usage"].get("completion_tokens", 0); ptoks = d["usage"].get("prompt_tokens", 0)
    t2 = time.time()
    return {"ttft": (t1 or t2) - t0, "tps": toks / (t2 - t1) if t1 and t2 > t1 else 0.0, "toks": toks, "ptoks": ptoks}


def prompt_for(length, task, k):
    if length == "court":
        return TASKS[task] % k
    salt = "Session %08x.\n\n" % random.getrandbits(32)
    return salt + CTX[length] + "\n\n" + TASKS_LONG[task] % k


def cell(length, task, n):
    res = []
    for rep in range(REPS):
        out = [None] * n; err = []
        def f(k):
            try: out[k] = one(prompt_for(length, task, rep * 10 + k), URLS[k % len(URLS)] if URLS else None)
            except Exception as e: err.append(str(e)[:120])
        th = [threading.Thread(target=f, args=(k,)) for k in range(n)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]
        if err: print("  error", err[0], flush=True); continue
        res.append((out, time.time() - t0))
    if not res:
        return {"bras": LABEL, "long": length, "type": task, "flux": n, "erreur": True}
    tps = [o["tps"] for out, _ in res for o in out]
    agg = [sum(o["toks"] for o in out) / wall for out, wall in res]
    ttft = [o["ttft"] for out, _ in res for o in out]
    # Field names kept as in the published data: bras = setup label, long = prompt length,
    # flux = requests at once, tps_total = total output tokens per second (median of reps).
    return {"bras": LABEL, "long": length, "type": task, "flux": n, "reps": len(res),
            "tps_flux": round(st.median(tps), 1), "tps_total": round(st.median(agg), 1),
            "tps_total_min": round(min(agg), 1), "tps_total_max": round(max(agg), 1),
            "ttft": round(st.median(ttft), 2), "ttft_max": round(max(ttft), 2),
            "jetons_prompt": max(o["ptoks"] for out, _ in res for o in out),
            "jetons_min": min(o["toks"] for out, _ in res for o in out), "heure": time.strftime("%H:%M %Z")}


if "--prose-en" in sys.argv:
    TASKS = {"prose EN": TASK_EN}; TASKS_LONG = {"prose EN": TASK_EN_LONG}

if __name__ == "__main__":
    one(next(iter(TASKS.values())) % 999)               # warm-up
    plan = [("court", t, n) for t in TASKS for n in (1, 4, 8)]
    if "--court-seul" not in sys.argv:
        plan += [(L, t, n) for L in ("16k", "64k") for t in TASKS for n in (1, 4, 8)]
    if "--flux" in sys.argv:
        fl = [int(x) for x in sys.argv[sys.argv.index("--flux") + 1].split(",")]
        lg = sys.argv[sys.argv.index("--longueurs") + 1].split(",") if "--longueurs" in sys.argv else ["court"]
        plan = [(L, t, n) for L in lg for t in TASKS for n in fl]
    for length, task, n in plan:
        r = cell(length, task, n)
        with open(RESULTS, "a") as w:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
        if r.get("erreur"):
            print("%-12s %-5s %-8s %d at once: ERROR" % (LABEL, length, task, n), flush=True); continue
        print("%-12s %-5s %-8s %d at once: %5.1f tok/s per request · %6.1f total [%s-%s] · first token %.2f s (max %.2f) · prompt %d · min %d tokens"
              % (LABEL, length, task, n, r["tps_flux"], r["tps_total"], r["tps_total_min"], r["tps_total_max"],
                 r["ttft"], r["ttft_max"], r["jetons_prompt"], r["jetons_min"]), flush=True)
    print("DONE", LABEL, time.strftime("%H:%M %Z"), flush=True)
