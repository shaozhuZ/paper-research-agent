"""Check the LLM judge against a human on a sample of answers.

    python eval/calibrate.py sheet --judged baseline      # writes eval/human_labels.csv
    (fill in human_correctness and human_faithful in Excel, save as CSV)
    python eval/calibrate.py compare --judged baseline

The sheet does not show the judge's scores, so the human labels aren't
anchored on them.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHEET = HERE / "human_labels.csv"
FIELDS = ["id", "question", "reference_answer", "answer", "retrieved_passages",
          "human_correctness", "human_faithful", "source_passage"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


# PDF extraction leaves NULs and other control characters in some chunks; csv can't write them
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    return _CONTROL.sub(" ", text)


def retrieved_by_id(label: str) -> dict[str, str]:
    """The passages the agent actually saw, so the human judges faithfulness on the same evidence as the LLM."""
    out = {}
    for r in load_jsonl(HERE / "results" / f"answers_{label}.jsonl"):
        if not r.get("error") and r.get("contexts"):
            out[r["id"]] = "\n\n".join(f"[{c['filename']} #{c['chunk_id']}]\n{c['content']}" for c in r["contexts"])
    return out


def make_sheet(args) -> None:
    judged = [r for r in load_jsonl(HERE / "results" / f"judged_{args.judged}.jsonl") if r["answered"]]
    questions = {q["id"]: q for q in load_jsonl(HERE / "questions_single_hop.jsonl")}
    retrieved = retrieved_by_id(args.judged)
    # keep labels already filled in, so regenerating the sheet never loses work
    existing = {}
    if SHEET.exists():
        with SHEET.open(encoding="utf-8-sig", newline="") as fh:
            existing = {r["id"]: r for r in csv.DictReader(fh)}
    by_id = {r["id"]: r for r in judged}
    ids = [i for i in existing if i in by_id] or [r["id"] for r in random.Random(args.seed).sample(judged, min(args.n, len(judged)))]
    sample = [by_id[i] for i in ids]
    # utf-8-sig so Excel on Windows opens it with the right encoding
    rows = []
    for r in sorted(sample, key=lambda r: r["id"]):
        q = questions[r["id"]]
        old = existing.get(r["id"], {})
        rows.append({"id": r["id"], "question": q["question"], "reference_answer": q["reference_answer"],
                     "answer": r["answer"], "retrieved_passages": retrieved.get(r["id"], ""),
                     "human_correctness": old.get("human_correctness", ""),
                     "human_faithful": old.get("human_faithful", ""),
                     "source_passage": q.get("source_text", "")})
    # build every row before opening the file, so a failure can't wipe existing labels
    with SHEET.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows({k: _clean(v) if isinstance(v, str) else v for k, v in row.items()} for row in rows)
    print(f"wrote {len(sample)} rows to {SHEET}")
    print("fill human_correctness (0/1/2) and human_faithful (y/n), then run: calibrate.py compare")


def compare(args) -> None:
    judged = {r["id"]: r for r in load_jsonl(HERE / "results" / f"judged_{args.judged}.jsonl")}
    with SHEET.open(encoding="utf-8-sig", newline="") as fh:
        human = [r for r in csv.DictReader(fh) if r["human_correctness"].strip()]
    if not human:
        print("no labels filled in yet")
        return

    agree_c = agree_f = agree_pass = 0
    disagreements = []
    for h in human:
        j = judged[h["id"]]
        hc = int(h["human_correctness"])
        hf = h["human_faithful"].strip().lower() in ("y", "yes", "true", "1")
        h_pass, j_pass = hc == 2 and hf, j["correctness"] == 2 and bool(j["faithful"])
        agree_c += hc == j["correctness"]
        agree_f += hf == j["faithful"]
        agree_pass += h_pass == j_pass
        if hc != j["correctness"] or hf != j["faithful"]:
            disagreements.append((h["id"], hc, hf, j["correctness"], j["faithful"], j.get("reason", "")))

    n = len(human)
    print(f"{n} labelled answers")
    print(f"correctness exact agreement: {agree_c}/{n} = {agree_c / n:.0%}")
    print(f"faithful agreement:          {agree_f}/{n} = {agree_f / n:.0%}")
    print(f"pass/fail agreement:         {agree_pass}/{n} = {agree_pass / n:.0%}")
    if disagreements:
        print("\ndisagreements (id  human c/f  judge c/f  judge reason):")
        for qid, hc, hf, jc, jf, reason in disagreements:
            print(f"  {qid}  {hc}/{'y' if hf else 'n'}  {jc}/{'y' if jf else 'n'}  {reason[:110]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("--judged", default="baseline")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("compare")
    c.add_argument("--judged", default="baseline")
    args = ap.parse_args()
    make_sheet(args) if args.cmd == "sheet" else compare(args)


if __name__ == "__main__":
    main()
