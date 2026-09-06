from agentic.graph import build_graph

app = build_graph()

result = app.invoke({"question": "What is climate change?"})

print("answer", result["answer"])
print("score:", result["faithfulness_score"])
print("attempts:", result["attempt_count"])
print("\history:")
for h in result["history"]:
    print(h)