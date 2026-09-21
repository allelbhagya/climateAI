# climateRAG

```bash
pip install -r requirements.txt

python -m agentic.test_graph 
```

```bash
uvicorn main:app --reload --port 8000

curl http://localhost:8000/ping

curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is climate change?"}'
```