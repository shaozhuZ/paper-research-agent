"""Measure end-to-end latency of POST /invoke.

Runs a fixed query set so before/after numbers are comparable.

Usage:
    python scripts/bench_latency.py --n 30 --concurrency 1 --label before
    python scripts/bench_latency.py --n 30 --concurrency 4 --label after-c4

Writes bench_results/<label>.json and prints p50/p95.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

QUERIES = [
    ("How does speculative sampling speed up LLM decoding?", "AI"),
    ("What is knowledge distillation and why does temperature matter?", "AI"),
    ("How does PagedAttention manage KV cache memory?", "AI"),
    ("What security weaknesses were found in GitHub Copilot generated code?", "Security"),
    ("Why do obfuscated gradients give a false sense of security?", "Security"),
    ("How can an off-path attacker infer TCP sequence numbers?", "Security"),
    ("What effects does mindfulness meditation have on immune function?", "Other"),
    ("What drives housing price dynamics across countries?", "Other"),
    ("Compare EfficientNet compound scaling with plain depth scaling.", "AI"),
    ("What role does renewable energy play in the global energy transition?", "All"),
]


def one(api: str, i: int) -> dict:
    q, domain = QUERIES[i % len(QUERIES)]
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{api}/invoke",
            json={"query": q, "language": "English", "domain": domain},
            timeout=300,
        )
        ok = r.ok
        err = None if ok else f"{r.status_code} {r.text[:200]}"
    except Exception as e:  # network error, timeout
        ok, err = False, repr(e)
    return {"i": i, "query": q, "ok": ok, "latency_s": time.perf_counter() - t0, "error": err}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()

    print("warmup ...")
    one(args.api, 0)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda i: one(args.api, i), range(args.n)))
    wall = time.perf_counter() - t0

    lat = [r["latency_s"] for r in rows if r["ok"]]
    summary = {
        "label": args.label,
        "n": args.n,
        "concurrency": args.concurrency,
        "errors": sum(not r["ok"] for r in rows),
        "p50_s": round(pct(lat, 50), 2) if lat else None,
        "p95_s": round(pct(lat, 95), 2) if lat else None,
        "mean_s": round(statistics.mean(lat), 2) if lat else None,
        "throughput_rps": round(len(lat) / wall, 3),
    }
    print(json.dumps(summary, indent=2))

    out = Path("bench_results")
    out.mkdir(exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
