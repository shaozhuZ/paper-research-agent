"""Upload every PDF under papers/<Domain>/ to the running API.

Layout (same as index/milvus_index.py):
    papers/AI/*.pdf
    papers/Security/*.pdf
    papers/Other/*.pdf

Usage:
    python scripts/ingest_papers.py --dir papers --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

DOMAINS = ("AI", "Security", "Other")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="papers")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"{root} not found", file=sys.stderr)
        return 1

    failed = 0
    for domain in DOMAINS:
        for pdf in sorted((root / domain).glob("*.pdf")):
            t0 = time.perf_counter()
            with pdf.open("rb") as fh:
                resp = requests.post(
                    f"{args.api}/upload",
                    files={"file": (pdf.name, fh, "application/pdf")},
                    data={"domain": domain},
                    timeout=300,
                )
            dt = time.perf_counter() - t0
            ok = resp.ok and "error" not in resp.text.lower()[:200]
            failed += 0 if ok else 1
            print(f"[{'ok' if ok else 'FAIL'}] {domain:8s} {pdf.name[:60]:60s} {dt:6.1f}s")
            if not ok:
                print("   ", resp.status_code, resp.text[:300])

    print("stats:", requests.get(f"{args.api}/stats", timeout=30).json())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
