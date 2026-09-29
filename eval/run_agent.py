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

# List prices, USD per 1M tokens: (input, cached input, output).
# Reasoning/thinking tokens are billed as output by both providers.
PRICES = {
    "google:gemini-3-flash-preview": (0.50, 0.05, 3.00),
    "openai:gpt-6-luna": (0.10, 0.01, 0.50),
}


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
        "usage": body.get("usage", {}),
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
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke runs)")
    ap.add_argument("--summary-only", action="store_true", help="print the summary of an existing run")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(args.questions, encoding="utf-8") if l.strip()]
    if args.only_failed_from:
        prev = HERE / "results" / f"answers_{args.only_failed_from}.jsonl"
        failed = {r["id"] for r in map(json.loads, prev.open(encoding="utf-8")) if r["error"]}
        questions = [q for q in questions if q["id"] in failed]
    if args.limit:
        questions = questions[: args.limit]
    out_path = HERE / "results" / f"answers_{args.label}.jsonl"
    if args.summary_only:
        summarize(load_latest(out_path), total=len(questions))
        return
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

    wanted = {q["id"] for q in questions}
    summarize([r for r in load_latest(out_path) if r["id"] in wanted], total=len(questions))


def load_latest(path: Path) -> list[dict]:
    """One row per question: the latest success if there is one, else the latest error."""
    best: dict[str, dict] = {}
    for r in map(json.loads, filter(str.strip, path.open(encoding="utf-8"))):
        prev = best.get(r["id"])
        if prev is None or not r["error"] or prev["error"]:
            best[r["id"]] = r
    return list(best.values())


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(int(len(s) * p), len(s) - 1)]


def cost_usd(u: dict) -> float | None:
    price = PRICES.get(u.get("model", "google:gemini-3-flash-preview"))
    if price is None:
        return None
    p_in, p_cached, p_out = price
    cached = u.get("cached_tokens", 0)
    # output_tokens already includes reasoning tokens (reasoning is a breakdown, not an extra)
    return ((u.get("input_tokens", 0) - cached) * p_in + cached * p_cached
            + u.get("output_tokens", 0) * p_out) / 1e6


def summarize(rows: list[dict], total: int) -> None:
    ok = [r for r in rows if not r["error"]]
    print(f"\n== answered {len(ok)}/{total} ==")
    if not ok:
        return
    lat = [r["latency_s"] for r in ok]
    print(f"latency   p50 {pct(lat, 0.5):6.1f}s   p95 {pct(lat, 0.95):6.1f}s")

    # older runs have no usage field
    used = [r["usage"] for r in ok if r.get("usage")]
    if not used:
        print("no usage data in these rows")
        return
    n = len(used)

    def avg(key: str) -> float:
        return sum(u.get(key, 0) for u in used) / n

    tool_calls = sum(len(u.get("tool_calls", [])) for u in used) / n
    llm_share = sum(u.get("llm_ms", 0) for u in used) / max(1, sum(u.get("latency_ms", 0) for u in used))
    models = sorted({u.get("model", "google:gemini-3-flash-preview") for u in used})
    modes = sorted({u.get("mode", "loop") for u in used})
    print(f"per question (n={n}, model {', '.join(models)}, mode {', '.join(modes)}):")
    print(f"  llm turns {avg('llm_turns'):.1f}   tool calls {tool_calls:.1f}")
    print(f"  tokens in {avg('input_tokens'):,.0f} (cached {avg('cached_tokens'):,.0f})"
          f"   out {avg('output_tokens'):,.0f}   of which reasoning {avg('reasoning_tokens'):,.0f}")
    costs = [cost_usd(u) for u in used]
    if None in costs:
        print(f"  time in LLM {llm_share:.0%}   cost: no price listed for {models}")
    else:
        cost = sum(costs) / n
        print(f"  time in LLM {llm_share:.0%}   cost ${cost:.4f}  (x{total} = ${cost * total:.2f})")


if __name__ == "__main__":
    main()
