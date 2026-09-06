from agentic.nodes import retrieve_node, generate_node, evaluate_node

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

eval_result = evaluate_node(state)
state["faithfulness_score"] = eval_result["faithfulness_score"]
state["history"] = state.get("history", []) + eval_result["history"] 

print("faithfulness score:", state["faithfulness_score"])
print("history:", state["history"])