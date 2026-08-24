import operator
from typing import TypedDict, Annotated

class AgentState(TypedDict):

    question: str
    retrieved_chunks: list
    answer:str
    faithfull_score: float
    attempt_count: int

    history: Annotated[list, operator.add]