from agentic.nodes import retrieve_node

state = {
    "question": "What is climate change?"
}

result = retrieve_node(state)
print(result)