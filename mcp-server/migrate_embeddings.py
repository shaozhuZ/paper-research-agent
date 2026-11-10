"""Re-embed a frozen Milvus corpus into a new collection; never drop a collection.

Uses the ordinary Voyage API and persists each response before inserting it.
Resume requires matching source hashes, model identity and cached target vectors.
Run inside mcp-server, with a writable checkpoint directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from embeddings import EmbeddingConfig, VoyageEmbeddings, validate_collection
from mcp_tool_server import _build_index_params, get_client
from retrieval import create_collection

FIELDS = ("filename", "chunk_id", "domain", "text")


def key(row: dict) -> tuple[str, int]:
    return row["filename"], int(row["chunk_id"])


def digest(rows: list[dict]) -> str:
    data = [{f: row[f] for f in FIELDS} for row in sorted(rows, key=key)]
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def snapshot(client, collection: str, vectors: bool = False) -> list[dict]:
    client.load_collection(collection)
    iterator = client.query_iterator(collection, batch_size=256, filter="",
                                     output_fields=[*FIELDS, *(["vector"] if vectors else [])],
                                     consistency_level="Strong")
    rows = []
    try:
        while batch := iterator.next():
            rows.extend(batch)
    finally:
        iterator.close()
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError(f"Duplicate filename/chunk_id in {collection}")
    return sorted(rows, key=key)


def save(path: Path, obj: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_expected(rows: list[dict], path: Path) -> str:
    expected = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(expected) != len(rows) or {key(row) for row in expected} != {key(row) for row in rows}:
        raise ValueError("Source IDs/count differ from the frozen evaluation corpus")
    lookup = {key(row): row for row in expected}
    for row in rows:
        exp = lookup[key(row)]
        if row["domain"] != exp["domain"] or row["text"].replace("\r\n", "\n").strip() != exp["text"].replace("\r\n", "\n").strip():
            raise ValueError(f"Frozen corpus content mismatch: {key(row)}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-corpus", type=Path, required=True)
    parser.add_argument("--model", default="voyage-4")
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.source == args.target:
        raise SystemExit("Source and target must be separate collections")
    if not 1 <= args.batch_size <= 128:
        raise SystemExit("batch-size must be 1..128")
    client = get_client()
    rows = snapshot(client, args.source)
    if not rows:
        raise SystemExit("Source collection is empty")
    corpus_hash = verify_expected(rows, args.expected_corpus)
    config = EmbeddingConfig("voyage", args.model, args.dimension)
    identity = {"source": args.source, "target": args.target, "source_sha256": digest(rows),
                "expected_corpus_sha256": corpus_hash, "embedding": config.metadata(args.dimension),
                "batch_size": args.batch_size, "input_type": "document", "output_dtype": "float",
                "truncation": False, "chunks": len(rows)}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.checkpoint_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise SystemExit("Checkpoint belongs to a different corpus/model/target")
    else:
        if client.has_collection(args.target):
            raise SystemExit("Target already exists without this checkpoint; choose a new collection")
        manifest = {"identity": identity, "started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
        save(manifest_path, manifest)
        (args.checkpoint_dir / "source_snapshot.jsonl").write_text(
            "".join(json.dumps({f: r[f] for f in FIELDS}, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    if not client.has_collection(args.target):
        index_params, _ = _build_index_params()
        create_collection(client, args.target, args.dimension, index_params, config.metadata(args.dimension))
    validate_collection(config, client.describe_collection(args.target))
    existing = {key(row): row for row in snapshot(client, args.target, vectors=True)}
    source = {key(row): row for row in rows}
    if not set(existing) <= set(source):
        raise SystemExit("Target contains unmanaged chunk IDs")
    for k, row in existing.items():
        if any(row[f] != source[k][f] for f in FIELDS):
            raise SystemExit("Target content differs from the source")
    embeddings = VoyageEmbeddings(args.model, args.dimension)
    tokens, requests, inserted = 0, 0, 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        cache_path = args.checkpoint_dir / f"batch_{start:06d}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            if any(key(row) in existing for row in batch):
                raise SystemExit("Target rows lack a saved embedding response; cannot verify their model")
            before = embeddings.total_tokens
            vectors = embeddings.embed_documents([row["text"] for row in batch])
            cached = {"identity": identity, "start": start, "rows_sha256": digest(batch),
                      "vectors": vectors, "total_tokens": embeddings.total_tokens - before}
            save(cache_path, cached)
        if cached["identity"] != identity or cached["start"] != start or cached["rows_sha256"] != digest(batch):
            raise SystemExit("Cached embeddings belong to different inputs")
        vectors = cached["vectors"]
        if len(vectors) != len(batch) or any(len(v) != args.dimension or not all(math.isfinite(x) for x in v) for v in vectors):
            raise SystemExit("Cached embedding vectors are invalid")
        pending = []
        for row, vector in zip(batch, vectors):
            if key(row) in existing:
                stored = existing[key(row)]["vector"]
                if len(stored) != len(vector) or any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(stored, vector)):
                    raise SystemExit("Target vector does not match saved Voyage response")
            else:
                pending.append({**{f: row[f] for f in FIELDS}, "vector": vector})
        if pending:
            result = client.insert(args.target, pending)
            if result["insert_count"] != len(pending):
                raise SystemExit("Milvus did not insert the complete batch; rerun from this checkpoint")
            inserted += len(pending)
        tokens += cached["total_tokens"]
        requests += 1
        manifest.update({"cached_batches": requests, "embedding_tokens": tokens,
                         "processed_chunks": start + len(batch), "status": "running"})
        save(manifest_path, manifest)
        print(f"embedded/verified {start + len(batch)}/{len(rows)}; newly inserted {inserted}; tokens {tokens}", flush=True)
    client.flush(args.target)
    final_rows = snapshot(client, args.target)
    current_source = snapshot(client, args.source)
    if digest(final_rows) != identity["source_sha256"] or digest(current_source) != identity["source_sha256"]:
        raise SystemExit("Source/target text metadata differ after migration")
    papers = {(row["domain"], row["filename"]) for row in rows}
    manifest.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                     "source_rows": len(current_source), "target_rows": len(final_rows),
                     "target_text_sha256": digest(final_rows), "papers_by_domain": dict(Counter(d for d, _ in papers)),
                     "embedding_tokens": tokens, "cached_batches": requests,
                     "api_tokens_this_invocation": embeddings.total_tokens,
                     "api_requests_this_invocation": embeddings.successful_requests})
    save(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ["status", "source_rows", "target_rows", "papers_by_domain", "embedding_tokens", "api_requests_this_invocation"]}), flush=True)


if __name__ == "__main__":
    main()
