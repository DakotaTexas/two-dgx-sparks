#!/usr/bin/env python3
"""Score table: accuracy per file (with Wilson 95 % interval), chance-corrected accuracy,
ECE, latency p50 / p95. Reads runs/<label>/<file>.jsonl written by run_jevbench.sh.

    python3 score.py <label> [<label> ...]
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = ("easy", "original", "hard")


def taches(tier):
    with open(os.path.join(HERE, "jevbench/datasets/public/%s.jsonl" % tier)) as f:
        return {t["id"]: t for t in map(json.loads, f)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - m, c + m)


def bras(nom):
    lignes, tout = [], []
    for tier in TIERS:
        p = os.path.join(HERE, "runs", nom, tier + ".jsonl")
        if not os.path.exists(p):
            lignes.append((tier, None))
            continue
        T = taches(tier)
        der = {}
        for r in map(json.loads, open(p)):
            der[r["task_id"]] = r                    # last attempt
        rs = [r for r in der.values() if r["task_id"] in T]
        ok = [r for r in rs if r.get("valid")]
        k = sum(1 for r in ok if r.get("correct"))
        ch = sum(1 / len(T[r["task_id"]]["labels"]) for r in rs) / max(len(rs), 1)
        acc = k / max(len(rs), 1)
        lignes.append((tier, dict(n=len(rs), invalides=len(rs) - len(ok), acc=acc,
                                  ic=wilson(k, len(rs)), corr=(acc - ch) / (1 - ch))))
        tout += [(r, T[r["task_id"]]) for r in ok]
    # top-label ECE (10 bins) and latency over all files
    cases = [[0, 0, 0] for _ in range(10)]
    lat = sorted(r["latency_s"] for r, _ in tout if r.get("latency_s") is not None)
    for r, t in tout:
        probs = r.get("probs") or {}
        if not probs:
            continue
        conf = max(probs.values())
        b = min(int(conf * 10), 9)
        cases[b][0] += 1
        cases[b][1] += conf
        cases[b][2] += 1 if r.get("correct") else 0
    n = sum(c[0] for c in cases) or 1
    ece = sum(abs(c[1] - c[2]) for c in cases if c[0]) / n
    q = lambda p: lat[min(int(p * len(lat)), len(lat) - 1)] if lat else float("nan")
    return lignes, ece, q(0.5), q(0.95)


def main():
    noms = sys.argv[1:] or sorted(os.listdir(os.path.join(HERE, "runs")))
    print("%-12s %-9s %4s %6s %13s %7s %4s" % ("label", "file", "n", "exact", "CI95", "corrected", "invalid"))
    for nom in noms:
        lignes, ece, p50, p95 = bras(nom)
        for tier, s in lignes:
            if s is None:
                print("%-12s %-9s  (not run)" % (nom, tier))
                continue
            print("%-12s %-9s %4d %5.1f%% %5.1f-%5.1f%% %6.1f%% %4d" % (
                nom, tier, s["n"], 100 * s["acc"], 100 * s["ic"][0], 100 * s["ic"][1],
                100 * s["corr"], s["invalides"]))
        print("%-12s ECE %.3f · latency p50 %.2f s · p95 %.2f s\n" % (nom, ece, p50, p95))


if __name__ == "__main__":
    main()
