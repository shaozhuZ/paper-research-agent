"""PDF -> chunks. Used by index_paper and by the eval scripts.

Keep this the only place that decides how papers are split. The eval set
stores gold answers as (filename, chunk_id); if indexing and eval ever split
differently, those ids point at different text and every score is wrong.
"""
from __future__ import annotations

import io
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def split_pdf(pdf_bytes: bytes, filename: str, domain: str) -> list[Document]:
    text = extract_text(pdf_bytes)
    if not text:
        raise ValueError(f"Could not extract text from {filename}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    doc = Document(page_content=text, metadata={"filename": filename, "domain": domain})
    chunks = splitter.split_documents([doc])
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
