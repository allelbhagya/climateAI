from base.query import search


def retrieve_node(state):
    question = state["question"]
    chunks = search(question, top_k=5)

    return {
        "rerived_chunks": chunks
    }