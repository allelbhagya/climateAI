import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
store_path = BASE_DIR / "storage"

index_path = store_path / "climate_index.faiss"
chunks_path = store_path / "chunks.json"

model = SentenceTransformer("all-MiniLM-L6-v2")

# reading index faiss and chunks file

def load_index():
    index = faiss.read_index(str(index_path))

    with open(chunks_path, "r") as f:
        chunks = json.load(f)

    return index, chunks

# input: query raw
# process: query -> encode -> search closest chunks with top k -> store {rank, distance, page, text}
def search(query, top_k = 3):

    index, chunks = load_index()

    query_vector = model.encode([query])
    distance, indices = index.search(query_vector, top_k)

    result = []

    for rank, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        result.append({
            "rank": rank+1,
            "distance": float(distance[0][rank]),
            "page": chunk["page"],
            "text": chunk["text"]
        })
    return result


if __name__ == "__main__":
    query = input("question: ")
    results = search(query, top_k=3)

    for r in results:
        print(f"\nrank: {r['rank']} page: {r['page']} distance: {r['distance']:.4f}")
        print(r["text"])