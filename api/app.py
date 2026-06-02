"""
api/app.py
----------
Streaming FastAPI service for the RAG system.
Deploy to AWS Lambda (via Mangum) or run locally.

Run locally:
    uvicorn api.app:app --reload --port 8000

Test:
    curl -X POST http://localhost:8000/ask \
      -H "Content-Type: application/json" \
      -d '{"question": "What is the capital of France?"}'
"""

import asyncio
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mangum import Mangum

app = FastAPI(title="RAG Production System", version="1.0.0")

# Lazy-load chain at first request (Lambda cold start friendly)
_chain = None
_retriever = None


def get_chain():
    global _chain, _retriever
    if _chain is None:
        from src.rag_chain import build_chain
        _chain, _retriever = build_chain()
    return _chain, _retriever


class Query(BaseModel):
    question: str
    stream: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(query: Query):
    chain, retriever = get_chain()

    if query.stream:
        async def generate():
            start = time.perf_counter()
            answer = chain.invoke({"question": query.question})
            latency_ms = (time.perf_counter() - start) * 1000

            for word in answer.split():
                yield word + " "
                await asyncio.sleep(0.015)  # simulate streaming

        return StreamingResponse(generate(), media_type="text/plain")
    else:
        start = time.perf_counter()
        answer = chain.invoke({"question": query.question})
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "question": query.question,
            "answer": answer,
            "latency_ms": round(latency_ms, 1),
        }


# AWS Lambda handler
handler = Mangum(app)
