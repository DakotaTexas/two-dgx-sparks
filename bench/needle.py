#!/usr/bin/env python3
"""Needle in a haystack at 200k / 500k / 800k tokens, 3 depths (10 / 50 / 90 %).

    python3 needle.py <chat_completions_url> <model>

Reasoning OFF, temperature 0. Pass = the 6-digit code is in the reply. Prints prompt_tokens and
total time. The haystack is the context file (BENCH_CONTEXT, default contexte.txt next to this
script, built by get_context.sh), repeated. CPT = characters per token, measured on that file
(240,000 characters -> 80,576 tokens). With another file, measure CPT again.
"""
import json, os, random, sys, time, urllib.request
URL, MODEL = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = open(os.environ.get("BENCH_CONTEXT", os.path.join(HERE, "contexte.txt"))).read()
CPT = 3.0
for target in (200_000, 500_000, 800_000):
    for depth in (0.1, 0.5, 0.9):
        code = "%06d" % random.randrange(10**6)
        n = int(target * CPT); hay = (RAW * (n // len(RAW) + 1))[:n]
        i = int(len(hay) * depth); i = hay.rfind("\n", 0, i) + 1
        text = hay[:i] + "\nIMPORTANT: the secret code of the Spark bench is %s.\n" % code + hay[i:]
        body = {"model": MODEL, "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Session %08x.\n\n" % random.getrandbits(32) + text +
                              "\n\nWhat is the secret code of the Spark bench? Answer with the 6 digits only."}]}
        t0 = time.time()
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=3600))
            reply = r["choices"][0]["message"]["content"] or ""; pt = r["usage"]["prompt_tokens"]
            ok = code in reply
            print("needle %4dk depth %d%%: %s · %d tokens · %.0f s · reply %r" % (target // 1000, depth * 100, "OK" if ok else "MISS", pt, time.time() - t0, reply[:30]), flush=True)
        except Exception as e:
            print("needle %4dk depth %d%%: ERROR %s (%.0f s)" % (target // 1000, depth * 100, str(e)[:150], time.time() - t0), flush=True)
print("DONE", time.strftime("%H:%M %Z"), flush=True)
