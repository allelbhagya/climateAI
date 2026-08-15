from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":

    text = "rising global tempreatures are accelerating ploar ice melt"

    vector = model.encode(text)
    print(vector)
    print(vector.shape)