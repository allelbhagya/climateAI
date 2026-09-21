import faiss
import json
from pathlib import Path

from .read_pdf import load_pdf
from .chunker import chunk_text
from .faiss_test import embedd_chunks, build_faiss_index

BASE_DIR = Path(__file__).resolve().parent
STORAGE_ROOT = BASE_DIR / "storage"


def build_pipeline(pdf_path, doc_id: str, chunk_size=600, overlap=50):
    doc_dir = STORAGE_ROOT / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    index_path = doc_dir / "index.faiss"
    chunks_path = doc_dir / "chunks.json"

    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
    embedd = embedd_chunks(chunks)
    index = build_faiss_index(embedd)

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w") as f:
        json.dump(chunks, f)

    print(f"[{doc_id}] indexed {len(chunks)} chunks")
    return len(chunks)