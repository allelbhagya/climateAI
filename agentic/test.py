from agentic.nodes import retrieve_node

from agentic.nodes import retrieve_node, generate_node

state = {
    "question": "What is climate change?"
}

retrieve_result = retrieve_node(state)
state.update(retrieve_result) 

print("retrieved chunks:", len(state["retrieved_chunks"]))

generate_result = generate_node(state)
state.update(generate_result)

print("attempt count:", state["attempt_count"])
print("answer:", state["answer"])