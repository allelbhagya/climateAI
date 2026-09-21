# climateRAG

- self-correcting rag agent with langgraph, cyclic graph scoring answer faithfulness at runtime and retries generation with a stricter prompt if claims aren't supported
- lightweight embedding-based faithfulness scorer instead of llm-as-judge, keeps the retry loop (max 3 attempts)
- retrieval -> generation (ollama cloud, gpt-oss:20b) -> faithfulness check -> retry/accept/give up loop
- keeps track of the best scoring attempt so a give-up still returns the strongest answer, not just the last one
- wrapped it in fastapi with /ask endpoint

setup

```bash
pip install -r requirements.txt

python -m agentic.test_graph 
```

test api: running streamlit

```bash
# terminal 1
uvicorn main:app --reload --port 8000

#terminal 2
streamlit run main_app.py
```