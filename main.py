from fastapi import FastAPI
from pydantic import BaseModel

from agentic.graph import build_graph

graph = build_graph()

app = FastAPI(title = "Climate QA")

class Question(BaseModel):
    question:str


@app.post("/ask")
def ask(q: Question):
    result = graph.invoke({"question":q.question})
    return {

        "answer": result["answer"],
        "score" : result["faithfulness_score"],
        "attempts" : result["attempt_count"],
        "history": result['history']
    }
