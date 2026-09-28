"""Copy chunks from the old dense-only collection into the hybrid one.

Reuses the stored embeddings, so nothing is sent to the embedding API. Run it
inside the mcp-server container, which already has the Milvus address:

    docker compose exec mcp-server python migrate_to_hybrid.py

Safe to re-run: it refuses to write into a target that already has rows
unless --drop is given.
"""
from __future__ import annotations

import argparse
import os

from pymilvus import DataType, MilvusClient

from mcp_tool_server import COLLECTION, MILVUS_TOKEN, MILVUS_URI, _build_index_params
from retrieval import create_collection

BATCH = 500


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.getenv("MILVUS_SOURCE_COLLECTION", "research_papers"))
    ap.add_argument("--target", default=COLLECTION)
    ap.add_argument("--drop", action="store_true", help="drop the target first if it exists")
    args = ap.parse_args()
    if args.source == args.target:
        raise SystemExit("source and target are the same collection")

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN or "")
    desc = client.describe_collection(args.source)
    fields = {f["name"]: f for f in desc["fields"]}
    vec_field = next(n for n, f in fields.items() if f["type"] == DataType.FLOAT_VECTOR)
    dim = int(fields[vec_field]["params"]["dim"])
    print(f"source {args.source}: fields {sorted(fields)}, vector field {vec_field!r}, dim {dim}")
    for needed in ("text", "filename", "domain", "chunk_id"):
        if needed not in fields:
            raise SystemExit(f"source has no {needed!r} field")

    if client.has_collection(args.target):
        if not args.drop:
            n = client.query(args.target, filter="", output_fields=["count(*)"])[0]["count(*)"]
            raise SystemExit(f"{args.target} already exists with {n} rows; pass --drop to rebuild it")
        client.drop_collection(args.target)
    index_params, _ = _build_index_params()
    create_collection(client, args.target, dim, index_params)

    client.load_collection(args.source)
    it = client.query_iterator(args.source, batch_size=BATCH, filter="",
                               output_fields=["text", vec_field, "filename", "domain", "chunk_id"])
    copied = 0
    while True:
        rows = it.next()
        if not rows:
            it.close()
            break
        client.insert(args.target, [
            {"text": r["text"], "vector": r[vec_field], "filename": r["filename"],
             "domain": r["domain"], "chunk_id": int(r["chunk_id"])}
            for r in rows
        ])
        copied += len(rows)
        print(f"copied {copied}")

    client.flush(args.target)
    n_src = client.query(args.source, filter="", output_fields=["count(*)"])[0]["count(*)"]
    n_dst = client.query(args.target, filter="", output_fields=["count(*)"])[0]["count(*)"]
    print(f"done: {n_src} rows in {args.source}, {n_dst} in {args.target}")
    if n_src != n_dst:
        raise SystemExit("row counts differ")


if __name__ == "__main__":
    main()
