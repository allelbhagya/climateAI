from langgraph.graph import StateGraph, END

from agentic.state import AgentState
from agentic.nodes import retrieve_node, generate_node, evaluate_node, finalize_node
from agentic.router import route_after_evaluate


def build_graph():
    graph = StateGraph(AgentState)

    # register nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("finalize", finalize_node)


    # entry point
    graph.set_entry_point("retrieve")

    # normal unconditional edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")

    # conditional edge
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "accept": END,
            "give_up": "finalize", # decide final
            "retry": "generate",   # loop back - retrieval
        }
    )

    return graph.compile()