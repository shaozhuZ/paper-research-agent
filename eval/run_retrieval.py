"""Run every question through vector_search and score the ranking.

Talks to the MCP tool server directly, so it measures retrieval on its own,
without the LLM in the loop.

    python eval/run_retrieval.py --questions eval/questions_single_hop.jsonl --label baseline
    python eval/run_retrieval.py --mode hybrid --label hybrid
    python eval/run_retrieval.py --mode hybrid --rerank --label hybrid_rerank

Without --mode it calls vector_search (whatever mode the server is set to);
with --mode it calls the retrieve tool, so modes can be compared side by side.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

from fastmcp import Client

from metrics import lenient_recall_at_k, lenient_reciprocal_rank, mean, recall_at_k, reciprocal_rank

HERE = Path(__file__).resolve().parent
KS = (1, 3, 5, 10)


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def unwrap(result) -> dict:
    """fastmcp returns structured content when the tool declares it, text otherwise."""
    data = getattr(result, "structured_content", None) or getattr(result, "data", None)
    if isinstance(data, dict):
        return data.get("result", data) if set(data) == {"result"} else data
    text = "".join(getattr(c, "text", "") for c in result.content)
    return json.loads(text)


def score(retrieved: list[tuple[str, int]], gold: set[tuple[str, int]]) -> dict:
    row = {f"recall@{k}": recall_at_k(retrieved, gold, k) for k in KS}
    row["rr"] = reciprocal_rank(retrieved, gold)
    # lenient: a neighbouring chunk also counts, since chunks overlap
    row.update({f"lenient_recall@{k}": lenient_recall_at_k(retrieved, gold, k) for k in KS})
    row["lenient_rr"] = lenient_reciprocal_rank(retrieved, gold)
    # paper level: did we at least land in the right paper?
    gold_papers = {f for f, _ in gold}
    row["paper_hit@5"] = (
        None if not gold_papers else float(any(f in gold_papers for f, _ in retrieved[:5]))
    )
    return row


def summarize(rows: list[dict]) -> dict:
    keys = ([f"recall@{k}" for k in KS] + ["rr", "paper_hit@5"]
            + [f"lenient_recall@{k}" for k in KS] + ["lenient_rr"])
    out = {k: mean([r[k] for r in rows]) for k in keys}
    out["mrr"] = out.pop("rr")
    out["lenient_mrr"] = out.pop("lenient_rr")
    out["n"] = len(rows)
    out["latency_p50_ms"] = sorted(r["latency_ms"] for r in rows)[len(rows) // 2] if rows else None
    return out


async def run(args) -> None:
    questions = load_questions(Path(args.questions))
    rows = []
    async with Client(args.mcp) as client:
        for q in questions:
            t0 = time.perf_counter()
            params = {"query": q["question"], "domain": args.domain or q["domain"], "top_k": max(KS)}
            if args.mode:
                res = unwrap(await client.call_tool("retrieve", {**params, "mode": args.mode, "rerank": args.rerank}))
                # the server falls back to first-stage order if rerank fails; that must not
                # end up in a table labelled "rerank"
                if args.rerank and not res.get("reranked"):
                    raise SystemExit(f"{q['id']}: rerank did not run (check VOYAGE_API_KEY and mcp-server logs)")
            else:
                res = unwrap(await client.call_tool("vector_search", params))
            latency_ms = round((time.perf_counter() - t0) * 1000)
            hits = res.get("results", [])
            retrieved = [(h["filename"], int(h["chunk_id"])) for h in hits]
            gold = {(g["filename"], int(g["chunk_id"])) for g in q["gold_chunks"]}
            row = {"id": q["id"], "domain": q["domain"], "type": q["type"],
                   "latency_ms": latency_ms, **score(retrieved, gold),
                   "retrieved": retrieved}
            rows.append(row)
            first = "miss" if not row["rr"] else f"rank {round(1 / row['rr'])}"
            print(f"{q['id']}  {first:8s} {q['question'][:70]}")

    overall = summarize(rows)
    by_domain = {d: summarize([r for r in rows if r["domain"] == d])
                 for d in sorted({r["domain"] for r in rows})}

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = {"label": args.label, "mode": args.mode or "server default", "rerank": args.rerank,
           "questions": args.questions,
           "domain_filter": args.domain or "per-question",
           "overall": overall, "by_domain": by_domain, "rows": rows}
    (out_dir / f"retrieval_{args.label}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n== {args.label} ({overall['n']} questions) ==")
    print(f"{'':10s}" + "".join(f"{k:>10s}" for k in ("R@1", "R@3", "R@5", "R@10", "MRR", "paper@5")))
    for name, s in [("overall", overall), *by_domain.items()]:
        vals = [s["recall@1"], s["recall@3"], s["recall@5"], s["recall@10"], s["mrr"], s["paper_hit@5"]]
        print(f"{name:10s}" + "".join(f"{v:10.3f}" for v in vals))
    print("lenient (neighbouring chunk also counts):")
    for name, s in [("overall", overall), *by_domain.items()]:
        vals = [s[f"lenient_recall@{k}"] for k in KS] + [s["lenient_mrr"]]
        print(f"{name:10s}" + "".join(f"{v:10.3f}" for v in vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(HERE / "questions_single_hop.jsonl"))
    ap.add_argument("--mcp", default="http://localhost:9000/mcp")
    ap.add_argument("--domain", default=None, help="force a domain filter, e.g. All")
    ap.add_argument("--label", default="run")
    ap.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default=None)
    ap.add_argument("--rerank", action="store_true", help="rerank the first-stage candidates (needs --mode)")
    args = ap.parse_args()
    if args.rerank and not args.mode:
        ap.error("--rerank needs --mode")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
