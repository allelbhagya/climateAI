import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from dotenv import load_dotenv
load_dotenv()

import requests
import numpy as np

from base.query import search, model as embedding_model

OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")


# reading question from state and returning retrived chunks
def retrieve_node(state):
    question = state["question"]
    chunks = search(question, top_k=5)

    return {"retrieved_chunks": chunks}


def build_context(chunks):
    context_parts = []
    for c in chunks:
        context_parts.append(f"[Page {c['page']}]\n{c['text']}")
    return "\n\n".join(context_parts)


# first attempt normal grounding
# any attempt_count >1, attempt strict instructions, including previous answer
def build_prompt(question, context, attempt_count, previous_answer=None):
    if attempt_count == 1:
        return f"""You are a climate science assistant. Answer the question using ONLY the context provided below.
                If the answer is not present in the context, say "I don't have enough information to answer that."

                Answer only using information explicitly supported by the retrieved passages.
                Do not generalize, infer, or add claims that are not directly supported.

                Context:
                {context}

                Question: {question}

                Answer:"""
    else:
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


# taking question, retrived chunks and attempt count from state
# return answera and attempt count (overwrite)
def generate_node(state):
    question = state["question"]
    chunks = state["retrieved_chunks"]
    attempt_count = state.get("attempt_count", 0) + 1

    context = build_context(chunks)
    previous_answer = state.get("answer")

    prompt = build_prompt(question, context, attempt_count, previous_answer)

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

def split_sentences(text):
    parts = text.replace("\n", " ").split(". ")
    return [p.strip() for p in parts if p.strip()]


def compute_faithfulness(answer, chunks):
    sentences = split_sentences(answer)
    if not sentences:
        return 0.0

    chunk_texts = [c["text"] for c in chunks]

    # embedd both sentences and chunks - return vector
    sentence_vectors = embedding_model.encode(sentences, convert_to_numpy=True)
    chunk_vectors = embedding_model.encode(chunk_texts, convert_to_numpy=True)
    # normalize both vectors l2 norm
    sentence_vectors = sentence_vectors / np.linalg.norm(sentence_vectors, axis=1, keepdims=True)
    chunk_vectors = chunk_vectors / np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    # cosine similarity 
    similarity_matrix = sentence_vectors @ chunk_vectors.T
    max_similarity_per_sentence = similarity_matrix.max(axis=1)

    return float(max_similarity_per_sentence.mean())


def evaluate_node(state):
    answer = state["answer"]
    chunks = state["retrieved_chunks"]
    attempt_count = state["attempt_count"]

    score = compute_faithfulness(answer, chunks)

    record = {
        "attempt": attempt_count,
        "answer": answer,
        "faithfulness_score": score,
    }

    update = {
        "faithfulness_score": score,
        "history": [record],
    }

    # keep best score

    if score > state.get("best_score", -1):
        update["best_score"] = score
        update["best_answer"] = answer

    return update

def finalize_node(state):
    if "best_answer" in state:
        return {
            "answer": state["best_answer"],
            "faithfulness_score": state["best_score"],
        }
    return {}