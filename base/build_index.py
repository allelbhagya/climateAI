# index and chunks file build

import os
import faiss
import json
from read_pdf import load_pdf # reading pdf as text indexed by pages
from chunker import chunk_text # loading recursive chunking
from faiss_test import embedd_chunks, build_faiss_index

path_store = "storage"
index_path = os.path.join(path_store, "climate_index.faiss")
chunks_path = os.path.join(path_store, "chunks.json")

# pdf -> {page, text} -> {chunks, page, text} -> {num_chunks, embeddings} -> {embedd_indexes}
def build_pipeline(pdf_path, chunk_size = 600, overlap = 50):

    os.makedirs(path_store, exist_ok=True)

    pages = load_pdf(pdf_path)
    chunks = chunk_text(pages, chunk_size=chunk_size, overlap=overlap)
    embedd = embedd_chunks(chunks)
    index = build_faiss_index(embedd)

    faiss.write_index(index, index_path)
    with open(chunks_path, "w") as f:
        json.dump(chunks,f)

    print("faiss db chunks:", len(chunks))

if __name__ == "__main__":
        path = "../data/climate_paper.pdf"
        build_pipeline(path)