from base.query import search
import requests
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

def retrieve_node(state):
    question = state["question"]
    chunks = search(question, top_k=5)

    return {
        "retrieved_chunks": chunks
    }

def build_context(chunks):
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Page {c['page']}]\n{c['text']}")
    return "\n\n".join(context_parts)

def build_prompt(question, context, attempt_count, previous_answer=None):
    if attempt_count == 1:
        # tier 1: baseline prompt
        return f"""You are a climate science assistant. Answer the question using ONLY the context provided below.
                    If the answer is not present in the context, say "I don't have enough information to answer that."

                    Answer only using information explicitly supported by the retrieved passages.
                    Do not generalize, infer, or add claims that are not directly supported.

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
    else:
        # tier 2/3 prompt
        return f"""You are a climate science assistant. Your previous answer was flagged as containing claims not directly supported by the context.

                    Previous answer:
                    {previous_answer}

                    Re-answer the question using ONLY information that is explicitly and directly stated in the context below.
                    Do not infer, generalize, combine facts, or add any claim you cannot point to a specific sentence for.
                    If something is not explicitly stated, say "I don't have enough information to answer that."

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""

def generate_node(state):
    question = state['question']
    chunks = state['retrieved_chunks']
    attempt_count = state.get("attempt_count", 0)+1

    context = build_context(chunks)
    previous_answer = state.get("answer")

    prompt =  build_prompt(question, context, attempt_count, previous_answer)
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
    answer = response.json()["message"]["content"].strip()

    return {
        "answer": answer,
        "attempt_count": attempt_count
    }

