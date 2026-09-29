"""Sample chunks and ask an LLM to write one question per chunk.

The chunk a question was written from becomes its gold chunk. The prompt
lives in eval/prompts/ so it can be edited without touching this file.

    python eval/generate_questions.py --n 60 --out eval/questions_draft.jsonl

For a held-out set, pass the existing question files to --exclude so no new
question is written from a chunk (or a neighbouring chunk) they already use:

    python eval/generate_questions.py --n 66 --seed 29 --id-prefix h \
        --exclude eval/questions_single_hop.jsonl --out eval/questions_heldout.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from llm import DEFAULT_EVAL_MODEL, get_llm

HERE = Path(__file__).resolve().parent
MIN_CHARS = 300  # very short chunks are usually headers, captions or page numbers
MAX_DIGIT_RATIO = 0.12  # above this a chunk is mostly a table, IPs or a reference list
COPY_NGRAM = 5  # a question sharing a 5-word run with its passage is copied, not paraphrased


def looks_like_table(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and sum(c.isdigit() for c in chars) / len(chars) > MAX_DIGIT_RATIO


def _words(text: str) -> list[str]:
    # PDF text breaks words across lines ("re- sponse"); join those back first
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def copied_span(question: str, passage: str, n: int = COPY_NGRAM) -> str | None:
    """Return the first n-word run the question shares with the passage, if any."""
    q, p = _words(question), _words(passage)
    grams = {tuple(p[i : i + n]) for i in range(len(p) - n + 1)}
    for i in range(len(q) - n + 1):
        if tuple(q[i : i + n]) in grams:
            return " ".join(q[i : i + n])
    return None


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def used_chunks(paths: list[str], margin: int = 1) -> set[tuple[str, int]]:
    """Gold chunks of existing questions, widened by `margin` on each side.

    Neighbours count as used because chunks overlap by 120 characters, so a
    question from the next chunk can have the same answer.
    """
    used = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                for g in json.loads(line)["gold_chunks"]:
                    for d in range(-margin, margin + 1):
                        used.add((g["filename"], g["chunk_id"] + d))
    return used


def sample_by_domain(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """Take roughly n/3 chunks from each domain so one domain can't dominate."""
    rng = random.Random(seed)
    by_domain = defaultdict(list)
    for c in chunks:
        if len(c["text"]) >= MIN_CHARS and not looks_like_table(c["text"]):
            by_domain[c["domain"]].append(c)
    per_domain = max(1, n // len(by_domain))
    picked = []
    for domain in sorted(by_domain):
        pool = by_domain[domain]
        picked += rng.sample(pool, min(per_domain, len(pool)))
    rng.shuffle(picked)
    return picked


def parse_json(text: str) -> dict | None:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(HERE / "data" / "chunks.jsonl"))
    ap.add_argument("--prompt", default=str(HERE / "prompts" / "gen_single_hop.txt"))
    ap.add_argument("--out", default=str(HERE / "questions_draft.jsonl"))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default=DEFAULT_EVAL_MODEL, help="provider:model")
    ap.add_argument("--exclude", nargs="*", default=[], help="question files whose gold chunks (and neighbours) are off limits")
    ap.add_argument("--id-prefix", default="q", help="use a different prefix for a held-out set so ids never clash")
    args = ap.parse_args()

    template = Path(args.prompt).read_text(encoding="utf-8")
    llm = get_llm(args.model, temperature=0.4)
    chunks = load_chunks(Path(args.chunks))
    if args.exclude:
        used = used_chunks(args.exclude)
        before = len(chunks)
        chunks = [c for c in chunks if (c["filename"], c["chunk_id"]) not in used]
        print(f"excluded {before - len(chunks)} chunks already used by {', '.join(args.exclude)}")
    picked = sample_by_domain(chunks, args.n, args.seed)

    kept = skipped = failed = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(picked, 1):
            prompt = template.format(filename=chunk["filename"], text=chunk["text"])
            try:
                reply = llm.invoke(prompt)
            except Exception as e:
                print(f"[{i}] LLM error: {e!r}")
                failed += 1
                time.sleep(2)
                continue
            obj = parse_json(reply.text if hasattr(reply, "text") else str(reply.content))
            if obj is None:
                failed += 1
                print(f"[{i}] could not parse reply")
                continue
            if obj.get("skip"):
                skipped += 1
                print(f"[{i}] skipped {chunk['filename']}#{chunk['chunk_id']}: {obj.get('reason', '')}")
                continue
            copied = copied_span(obj["question"], chunk["text"])
            if copied:
                skipped += 1
                print(f"[{i}] dropped, copies passage: '{copied}'")
                continue
            item = {
                "id": f"{args.id_prefix}{kept + 1:03d}",
                "question": obj["question"],
                "type": "single_hop",
                "domain": chunk["domain"],
                "language": "English",
                "answerable": True,
                "gold_chunks": [{"filename": chunk["filename"], "chunk_id": chunk["chunk_id"]}],
                "reference_answer": obj["reference_answer"],
                "source_text": chunk["text"],  # kept for manual review, dropped later
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
            print(f"[{i}] {item['id']}: {item['question']}")

    print(f"kept {kept}, skipped {skipped}, failed {failed} -> {args.out}")


if __name__ == "__main__":
    main()
