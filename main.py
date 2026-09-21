import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from agentic.graph import build_graph
from base.build_index import build_pipeline  # your refactored pipeline script

# Step 1: build the graph once, when the server starts.
graph = build_graph()

# Where uploaded PDFs get temporarily saved before indexing.
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Step 2: create the FastAPI app.
app = FastAPI(title="Climate QA")


# Simple sanity-check routes — no graph, no LLM calls.
# Use these to confirm the server itself is up before testing /ask.
@app.get("/")
def hello():
    return {"message": "hello"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


# Step 3: upload a PDF -> build a faiss index for it -> return a doc_id.
# The client keeps this doc_id and sends it along with every question.
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = build_pipeline(str(pdf_path), doc_id=doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}") from e

    return {"doc_id": doc_id, "chunks_indexed": num_chunks}


# Step 4: ask a question against a specific doc_id's index.
class Question(BaseModel):
    question: str
    doc_id: str


@app.post("/ask")
def ask(q: Question):
    result = graph.invoke({"question": q.question, "doc_id": q.doc_id})
    return {
        "answer": result["answer"],
        "score": result["faithfulness_score"],
        "attempts": result["attempt_count"],
        "history": result["history"],
    }