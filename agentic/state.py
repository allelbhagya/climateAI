import operator
from typing import TypedDict, Annotated, Optional

class AgentState(TypedDict):
    question: str
    doc_id: str 
    answer: Optional[str]
    retrieved_chunks: Optional[list]
    faithfulness_score: Optional[float]
    attempt_count: int
    history: list
    best_answer: Optional[str]
    best_score: Optional[float]

    history: Annotated[list, operator.add]