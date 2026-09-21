import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
STORAGE_ROOT = BASE_DIR / "storage"

model = SentenceTransformer("all-MiniLM-L6-v2")


def _paths_for(doc_id: str):
    doc_dir = STORAGE_ROOT / doc_id
    return doc_dir / "index.faiss", doc_dir / "chunks.json"


def load_index(doc_id: str):
    index_path, chunks_path = _paths_for(doc_id)
    if not index_path.exists():
        raise FileNotFoundError(f"No index found for doc_id={doc_id}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    return index, chunks


def search(query, doc_id: str, top_k=3):
    index, chunks = load_index(doc_id)

    query_vector = model.encode([query])
    distance, indices = index.search(query_vector, top_k)

    result = []
    for rank, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        result.append({
            "rank": rank + 1,
            "distance": float(distance[0][rank]),
            "page": chunk["page"],
            "text": chunk["text"]
        })
    return result