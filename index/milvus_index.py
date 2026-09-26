import os
import time
from pathlib import Path
from typing import List

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from pypdf import PdfReader

# ----------------------------
# Config (env-driven)
# ----------------------------
COLLECTION    = os.getenv("MILVUS_COLLECTION", "research_papers")
MILVUS_URI    = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN  = os.getenv("MILVUS_TOKEN", "")
INDEX_TYPE    = os.getenv("INDEX_TYPE", "HNSW").upper()  # HNSW, IVF_PQ, or DISKANN
EMB_MODEL     = os.getenv("GEMINI_EMB_MODEL", "gemini-embedding-001")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
PAPERS_DIR    = Path(os.getenv("PAPERS_DIR", "papers"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "50"))  # embedding batch size

# TODO: Update this with your Google AI Studio API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

DOMAINS = ["AI", "Security", "Other"]


# ----------------------------
# PDF loading
# ----------------------------
def load_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


# ----------------------------
# Milvus helpers
# ----------------------------
def connect_milvus() -> None:
    conn_args = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        conn_args["token"] = MILVUS_TOKEN
    connections.connect(alias="default", **conn_args)


def reset_collection() -> None:
    if utility.has_collection(COLLECTION):
        utility.drop_collection(COLLECTION)
        print(f"Dropped existing collection '{COLLECTION}'")


def create_collection() -> Collection:
    # Reference: https://milvus.io/api-reference/pymilvus/v2.4.x/ORM/FieldSchema/FieldSchema.md
    fields = [
        FieldSchema(name="id",        dtype=DataType.INT64,         is_primary=True, auto_id=True),
        FieldSchema(name="text",      dtype=DataType.VARCHAR,       max_length=65535),
        FieldSchema(name="filename",  dtype=DataType.VARCHAR,       max_length=512),
        FieldSchema(name="domain",    dtype=DataType.VARCHAR,       max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR,  dim=3072),
    ]
    schema = CollectionSchema(fields)
    collection = Collection(name=COLLECTION, schema=schema)
    print(f"Created collection '{COLLECTION}'")
    return collection


def build_index(collection: Collection) -> None:
    if INDEX_TYPE == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
    elif INDEX_TYPE == "IVF_PQ":
        index_params = {
            "index_type": "IVF_PQ",
            "metric_type": "COSINE",
            "params": {"nlist": 128, "m": 8, "nbits": 8},
        }
    elif INDEX_TYPE == "DISKANN":
        index_params = {
            "index_type": "DISKANN",
            "metric_type": "COSINE",
            "params": {},
        }
    else:
        raise ValueError(f"Unsupported INDEX_TYPE: {INDEX_TYPE}. Use HNSW, IVF_PQ, or DISKANN.")

    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Index built: {INDEX_TYPE}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY.")

    if not PAPERS_DIR.exists():
        raise RuntimeError(
            f"Papers directory '{PAPERS_DIR}' not found.\n"
            f"Expected: papers/AI/*.pdf, papers/Security/*.pdf, papers/Other/*.pdf"
        )

    print(f"INDEX_TYPE:  {INDEX_TYPE}")
    print(f"COLLECTION:  {COLLECTION}")
    print(f"MILVUS_URI:  {MILVUS_URI}\n")

    # Connect and reset
    connect_milvus()
    reset_collection()
    collection = create_collection()

    emb_model = GoogleGenerativeAIEmbeddings(model=EMB_MODEL)

    all_texts      = []
    all_filenames  = []
    all_domains    = []

    # ----------------------------
    # Load and chunk all PDFs
    # ----------------------------
    for domain in DOMAINS:
        domain_dir = PAPERS_DIR / domain
        if not domain_dir.exists():
            print(f"[{domain}] Directory not found, skipping.")
            continue

        pdf_files = sorted(domain_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[{domain}] No PDFs found.")
            continue

        print(f"[{domain}] {len(pdf_files)} papers found:")
        for pdf_path in pdf_files:
            print(f"  Loading: {pdf_path.name} ...", end=" ", flush=True)
            try:
                text = load_pdf(pdf_path)
                if not text:
                    print("SKIPPED (no text)")
                    continue
                chunks = chunk_text(text)
                all_texts.extend(chunks)
                all_filenames.extend([pdf_path.name] * len(chunks))
                all_domains.extend([domain] * len(chunks))
                print(f"{len(chunks)} chunks")
            except Exception as e:
                print(f"ERROR: {e}")
        print()

    if not all_texts:
        raise RuntimeError("No chunks to index.")

    print(f"Total chunks to embed: {len(all_texts)}")

    # ----------------------------
    # Generate embeddings in batches
    # ----------------------------
    print(f"\nGenerating embeddings (batch_size={BATCH_SIZE}) ...")
    all_embeddings = []
    embed_time = 0

    for i in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[i:i + BATCH_SIZE]
        retry = 0
        while retry < 3:
            try:
                start = time.time()
                embeddings = emb_model.embed_documents(batch)
                embed_time += time.time() - start
                all_embeddings.extend(embeddings)
                break
            except Exception as e:
                retry += 1
                if retry < 3:
                    print(f"  Error, retrying... ({retry}/3)")
                    time.sleep(60)
                else:
                    raise
        print(f"  Embedded {min(i + BATCH_SIZE, len(all_texts))}/{len(all_texts)}")
        if i + BATCH_SIZE < len(all_texts):
            time.sleep(1)

    print(f"Embedding done: {embed_time:.2f}s")

    # ----------------------------
    # Insert into Milvus
    # ----------------------------
    print(f"\nInserting {len(all_texts)} vectors into Milvus ...")
    insert_batch = 1000
    insert_start = time.time()

    for i in range(0, len(all_texts), insert_batch):
        end = min(i + insert_batch, len(all_texts))
        collection.insert([
            all_texts[i:end],
            all_filenames[i:end],
            all_domains[i:end],
            all_embeddings[i:end],
        ])
        print(f"  Inserted {end}/{len(all_texts)}")

    collection.flush()
    insert_time = time.time() - insert_start

    # ----------------------------
    # Build index
    # ----------------------------
    print("\nBuilding index ...")
    index_start = time.time()
    build_index(collection)
    collection.load()
    index_time = time.time() - index_start

    connections.disconnect(alias="default")

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Collection:      {COLLECTION}")
    print(f"Milvus URI:      {MILVUS_URI}")
    print(f"Index type:      {INDEX_TYPE}")
    print(f"Total chunks:    {len(all_texts)}")
    print(f"Embedding time:  {embed_time:.2f}s")
    print(f"Insert time:     {insert_time:.2f}s")
    print(f"Index time:      {index_time:.2f}s")
    print(f"Total time:      {embed_time + insert_time + index_time:.2f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()