"""Grade the agent's answers with an LLM judge.

Reads the answer files written by run_agent.py. When a question failed in the
first file but was answered in a later one (e.g. a retry with a higher step
limit), the later answer is used. Questions that never got an answer score 0
and count as not faithful-and-correct.

    python eval/judge_answers.py --answers v0 --label v0_j2 --prompt judge_v2
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm import DEFAULT_EVAL_MODEL, get_llm

HERE = Path(__file__).resolve().parent
NEIGHBOURS = 1


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def merge_answers(labels: list[str]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for label in labels:
        for row in load_jsonl(HERE / "results" / f"answers_{label}.jsonl"):
            prev = merged.get(row["id"])
            if prev is None or (prev["error"] and not row["error"]):
                merged[row["id"]] = row
    return merged


def source_passages(q: dict, chunks: dict[tuple[str, int], str]) -> str:
    parts = []
    for g in q["gold_chunks"]:
        for cid in range(g["chunk_id"] - NEIGHBOURS, g["chunk_id"] + NEIGHBOURS + 1):
            text = chunks.get((g["filename"], cid))
            if text:
                parts.append(f"[{g['filename']} #{cid}]\n{text}")
    return "\n\n".join(parts)


def retrieved_passages(answer: dict) -> str:
    return "\n\n".join(f"[{c['filename']} #{c['chunk_id']}]\n{c['content']}" for c in answer["contexts"])


def parse(text: str) -> dict | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if obj.get("correctness") not in (0, 1, 2) or not isinstance(obj.get("faithful"), bool):
        return None
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", required=True, help="answer labels, later ones fill failures")
    ap.add_argument("--questions", default=str(HERE / "questions_single_hop.jsonl"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default=DEFAULT_EVAL_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prompt", default="judge_v2", help="prompt file name under eval/prompts/")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions, to match run_agent.py --limit")
    args = ap.parse_args()

    questions = {q["id"]: q for q in load_jsonl(Path(args.questions))[: args.limit]}
    chunks = {(c["filename"], c["chunk_id"]): c["text"] for c in load_jsonl(HERE / "data" / "chunks.jsonl")}
    answers = merge_answers(args.answers)
    template = (HERE / "prompts" / f"{args.prompt}.txt").read_text(encoding="utf-8")
    llm = get_llm(args.model, temperature=0.0)

    def grade(qid: str) -> dict:
        q, a = questions[qid], answers.get(qid)
        base = {"id": qid, "domain": q["domain"], "answer": (a or {}).get("answer", ""),
                "latency_s": (a or {}).get("latency_s")}
        if a is None or a["error"] or not a["answer"].strip():
            return {**base, "answered": False, "correctness": 0, "faithful": None,
                    "reason": (a or {}).get("error") or "no answer"}
        # faithfulness is judged against what the agent retrieved; older runs didn't
        # record that, so fall back to the gold passage and its neighbours
        passages = retrieved_passages(a) if a.get("contexts") else source_passages(q, chunks)
        prompt = template.format(question=q["question"], reference_answer=q["reference_answer"],
                                 source_passages=passages, answer=a["answer"])
        for _ in range(3):
            reply = llm.invoke(prompt)
            obj = parse(reply.text if hasattr(reply, "text") else str(reply.content))
            if obj:
                return {**base, "answered": True, **obj}
        return {**base, "answered": True, "correctness": None, "faithful": None, "reason": "judge output unparseable"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(grade, sorted(questions)))

    out = HERE / "results" / f"judged_{args.label}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    answered = [r for r in rows if r["answered"]]
    graded = [r for r in answered if r["correctness"] is not None]
    passed = [r for r in graded if r["correctness"] == 2 and r["faithful"]]
    lat = sorted(r["latency_s"] for r in answered if r["latency_s"] is not None)
    print(f"\n== {args.label}: {n} questions, judge {args.model}, prompt {args.prompt} ==")
    print(f"no answer (loop / error):   {n - len(answered):4d}  ({(n - len(answered)) / n:.1%})")
    print(f"correct AND faithful:       {len(passed):4d}  ({len(passed) / n:.1%})   <- headline, over all questions")
    print(f"correctness 2 / 1 / 0:      " + " / ".join(str(sum(r['correctness'] == s for r in graded)) for s in (2, 1, 0)))
    print(f"unfaithful (answered):      {sum(r['faithful'] is False for r in graded):4d}  ({sum(r['faithful'] is False for r in graded) / max(1, len(graded)):.1%} of answered)")
    if lat:
        print(f"latency p50 / p95:          {lat[len(lat) // 2]:.1f}s / {lat[int(len(lat) * 0.95)]:.1f}s")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
