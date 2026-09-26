"""Upload every PDF under <dir>/<Domain>/ to the running API.

Safe to re-run: papers that already went in are recorded in data/ingested.txt
and skipped, so a run that dies halfway can just be started again. When the
embedding API is rate limited (429) the paper is retried after a pause.

    python scripts/ingest_papers.py --dir <papers dir> --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

DOMAINS = {"ai": "AI", "security": "Security", "other": "Other"}
DONE_FILE = Path("data/ingested.txt")


def is_rate_limited(resp: requests.Response) -> bool:
    return resp.status_code == 429 or "RESOURCE_EXHAUSTED" in resp.text or " 429" in resp.text


def upload(api: str, pdf: Path, domain: str) -> requests.Response:
    with pdf.open("rb") as fh:
        return requests.post(
            f"{api}/upload",
            files={"file": (pdf.name, fh, "application/pdf")},
            data={"domain": domain},
            timeout=600,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="papers")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--wait", type=float, default=65.0, help="seconds to wait after a 429")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"{root} not found", file=sys.stderr)
        return 1

    DONE_FILE.parent.mkdir(exist_ok=True)
    done = set(DONE_FILE.read_text(encoding="utf-8").splitlines()) if DONE_FILE.exists() else set()

    failed = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        domain = DOMAINS.get(folder.name.lower())
        if domain is None:
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            if pdf.name in done:
                print(f"[skip] {domain:8s} {pdf.name[:60]}")
                continue

            for attempt in range(1, args.retries + 1):
                t0 = time.perf_counter()
                resp = upload(args.api, pdf, domain)
                dt = time.perf_counter() - t0
                if resp.ok:
                    print(f"[ok]   {domain:8s} {pdf.name[:60]:60s} {dt:6.1f}s")
                    with DONE_FILE.open("a", encoding="utf-8") as fh:
                        fh.write(pdf.name + "\n")
                    break
                if is_rate_limited(resp) and attempt < args.retries:
                    print(f"[429]  {domain:8s} {pdf.name[:60]:60s} waiting {args.wait:.0f}s (try {attempt})")
                    time.sleep(args.wait)
                    continue
                print(f"[FAIL] {domain:8s} {pdf.name[:60]}\n       {resp.status_code} {resp.text[:300]}")
                failed.append(pdf.name)
                break

    print("stats:", requests.get(f"{args.api}/stats", timeout=30).json())
    if failed:
        print(f"{len(failed)} failed; run the same command again to retry them")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
