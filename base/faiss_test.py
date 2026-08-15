import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# input: dict of chunks {chunk_id, page, text}
# output: {num_chunks, embedding_dim}
def embedd_chunks(chunks):

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

# input: vector embeddings
# output: indexes 
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":

    from read_pdf import load_pdf
    from chunker import chunk_text

    path = "../data/climate_paper.pdf"
    pages = load_pdf(path)
    chunks = chunk_text(pages, chunk_size=600, overlap=50)

    embedd = embedd_chunks(chunks)
    print("embedd shape: ", embedd.shape)

    index = build_faiss_index(embedd)
    print("total vectors in faiss: ", index.ntotal)

# embedd shape:  (143, 384)
# total vectors in faiss:  143