import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

def build_context(chunks):
    # taking faiss retrived chunks {page, text, distance}
    context_parts = [] 
    for c in chunks:
        context_parts.append(f"[Page {c['page']}]\n{c['text']}")
    return "\n\n".join(context_parts)


def build_prompt(query, context):

    # addition to not genralize add claims -> low faithfullness 
    return f"""You are a climate science assistant. Answer the question using ONLY the context provided below.
    If the answer is not present in the context, say "I don't have enough information to answer that."
    
    Answer only using information explicitly supported by the retrieved passages. 
    Do not generalize, infer, or add claims that are not directly supported.

    Context:
    {context}

    Question: {query}

    Answer:"""

from query import search


# input: query
# process: search retrived chunks closest -> append {prompt + context + query}
def generate_answer(query, top_k=3):
    retrieved = search(query, top_k=top_k)
    context = build_context(retrieved)
    prompt = build_prompt(query, context)

    response = requests.post(
        OLLAMA_CLOUD_URL,
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )

    response.raise_for_status()
    answer = response.json()["message"]["content"]

    return {
        "query": query,
        "answer": answer.strip(),
        "contexts": [c["text"] for c in retrieved],
        "sources": [{"page": c["page"], "distance": c["distance"]} for c in retrieved]
    }

if __name__ == "__main__":
    print(generate_answer("What is climate change?"))