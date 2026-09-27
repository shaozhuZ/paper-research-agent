"""Send every question through the full agent (POST /invoke) and save the answers.

Answers are appended as they come back, so an interrupted run can be resumed:
questions already answered successfully are skipped, failed ones are retried.

    python eval/run_agent.py --label baseline
    python eval/run_agent.py --label retry25 --only-failed-from baseline
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests

HERE = Path(__file__).resolve().parent


def ask(api: str, q: dict, timeout: float) -> dict:
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{api}/invoke",
            json={"query": q["question"], "language": q.get("language", "English"), "domain": q["domain"]},
            timeout=timeout,
        )
        body = r.json() if r.ok else {}
        error = None if r.ok else f"{r.status_code} {r.text[:300]}"
    except Exception as e:
        body, error = {}, repr(e)
    return {
        "id": q["id"],
        "answer": body.get("answer", ""),
        "papers": body.get("papers", []),
        "recommended_papers": body.get("recommended_papers", []),
        "contexts": body.get("contexts", []),
        "latency_s": round(time.perf_counter() - t0, 2),
        "error": error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "questions_single_hop.jsonl"))
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--label", default="run")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--only-failed-from", metavar="LABEL",
                    help="only ask the questions that errored in answers_<LABEL>.jsonl")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    if args.only_failed_from:
        prev = HERE / "results" / f"answers_{args.only_failed_from}.jsonl"
        failed = {r["id"] for r in map(json.loads, prev.open(encoding="utf-8")) if r["error"]}
        questions = [q for q in questions if q["id"] in failed]
    out_path = HERE / "results" / f"answers_{args.label}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    done = set()
    if out_path.exists():
        # only successful answers count as done; failed ones are asked again and the
        # new line is appended (judge_answers.py prefers a later success over an error)
        done = {r["id"] for r in map(json.loads, filter(str.strip, out_path.open(encoding="utf-8")))
                if not r["error"]}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(done)} already answered, {len(todo)} to go -> {out_path}")

    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as fh:
        futures = [pool.submit(ask, args.api, q, args.timeout) for q in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            status = "ERR" if row["error"] else "ok "
            print(f"[{n}/{len(todo)}] {status} {row['id']} {row['latency_s']:6.1f}s  {row['answer'][:70]!r}")


if __name__ == "__main__":
    main()
