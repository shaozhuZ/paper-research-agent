"""Split every paper the same way index_paper does and dump the chunks.

Output: one JSON object per line with filename, domain, chunk_id, text.
Question generation samples from this file, so its (filename, chunk_id)
pairs line up with what is stored in Milvus.

    python eval/build_chunks.py --papers <papers dir> --out eval/data/chunks.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mcp-server"))
from chunking import split_pdf  # noqa: E402

# Folder names on disk vary in case ("Ai" vs "AI"); the index uses these.
DOMAINS = {"ai": "AI", "security": "Security", "other": "Other"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", required=True)
    ap.add_argument("--out", default="eval/data/chunks.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for folder in sorted(Path(args.papers).iterdir()):
            domain = DOMAINS.get(folder.name.lower())
            if not folder.is_dir() or domain is None:
                continue
            for pdf in sorted(folder.glob("*.pdf")):
                chunks = split_pdf(pdf.read_bytes(), pdf.name, domain)
                for c in chunks:
                    fh.write(json.dumps({
                        "filename": c.metadata["filename"],
                        "domain": domain,
                        "chunk_id": c.metadata["chunk_id"],
                        "text": c.page_content,
                    }, ensure_ascii=False) + "\n")
                total += len(chunks)
                print(f"{domain:8s} {len(chunks):4d}  {pdf.name}")
    print(f"total chunks: {total} -> {out}")


if __name__ == "__main__":
    main()
