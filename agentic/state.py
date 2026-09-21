import operator
from typing import TypedDict, Annotated, Optional

class AgentState(TypedDict):

    question: str
    retrieved_chunks: list
    answer:str
    faithfulness_score: float
    attempt_count: int
    best_answer: Optional[str]
    best_score: Optional[float]

    history: Annotated[list, operator.add]