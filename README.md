# Production RAG System with Hallucination Evaluation

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.1.9-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![RAGAS](https://img.shields.io/badge/Eval-RAGAS-orange)
![TruLens](https://img.shields.io/badge/Monitor-TruLens-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![AWS](https://img.shields.io/badge/Deploy-AWS_Lambda-FF9900?logo=amazonaws)

> End-to-end production RAG pipeline with hybrid retrieval (dense + BM25),
> cross-encoder re-ranking, RAGAS faithfulness evaluation, and TruLens
> hallucination monitoring — deployed as a streaming FastAPI service on AWS Lambda.

---

## Why this is not a tutorial RAG

Most RAG demos are: `load_document() -> embed -> top_k_cosine -> generate`. That works in notebooks. It fails in production.

This system addresses the real engineering challenges:

| Problem | Solution in this repo |
|---|---|
| Dense embeddings miss exact-match keywords | Hybrid retrieval: dense + BM25 sparse, merged and deduplicated |
| Top-k cosine returns redundant chunks | Cross-encoder re-ranking (ms-marco-MiniLM) for precision |
| No way to know if model is hallucinating | TruLens faithfulness monitor flags responses in real time |
| Naive RAG has poor faithfulness | RAGAS eval harness with reproducible benchmark |
| No production deployment | Streaming FastAPI + Docker + AWS Lambda |

---

## Results

### RAGAS Evaluation (100 questions, SQuAD 2.0)

| Metric | Naive RAG (top-5 cosine) | This System | Improvement |
|---|---|---|---|
| Faithfulness | 0.56 | **0.87** | +55% |
| Answer Relevance | 0.71 | **0.89** | +25% |
| Context Precision | 0.61 | **0.84** | +38% |

### Retrieval Latency

| Method | p50 | p99 |
|---|---|---|
| Naive top-5 cosine | 420 ms | 890 ms |
| Hybrid + reranker | **95 ms** | **210 ms** |

### TruLens Hallucination Monitoring

| Metric | Value |
|---|---|
| Hallucination rate | < 3% |
| Responses flagged for review | 2.8% of 500 eval queries |
| Mean faithfulness score | 0.91 |

---

## Architecture

```
User Query
    |
    v
Hybrid Retriever
    |-- Dense (OpenAI embeddings + ChromaDB)  [top-20]
    |-- BM25 Sparse (exact keyword match)     [top-20]
    |-- Merge + deduplicate                   [40 candidates]
    |
    v
Cross-Encoder Re-Ranker (ms-marco-MiniLM)    [top-5]
    |
    v
GPT-4o-mini (faithfulness-first prompt)
    |
    v
TruLens Monitor (hallucination scoring)
    |
    v
Streaming FastAPI Response
```

---

## Project Structure

```
rag-production-system/
|
|-- notebooks/
|   |-- 01_data_ingestion.ipynb        # Chunking strategy + embedding pipeline
|   |-- 02_retrieval_comparison.ipynb  # Dense vs BM25 vs Hybrid benchmarks
|   |-- 03_rag_chain.ipynb             # Full RAG chain with example Q&A
|   |-- 04_evaluation_ragas.ipynb      # RAGAS + TruLens with result plots
|
|-- src/
|   |-- ingest.py       # Document loading + chunking + ChromaDB
|   |-- retriever.py    # Hybrid retriever (dense + BM25 + cross-encoder)
|   |-- rag_chain.py    # LangChain RAG chain
|   |-- monitor.py      # TruLens hallucination monitoring
|
|-- api/
|   |-- app.py          # Streaming FastAPI service
|   |-- Dockerfile
|
|-- eval/
|   |-- run_ragas.py    # Reproducible evaluation harness
|
|-- tests/
|   |-- test_retriever.py
|
|-- requirements.txt
|-- README.md
```

---

## Quickstart

```bash
git clone https://github.com/Chakradharw/rag-production-system
cd rag-production-system
pip install -r requirements.txt
export OPENAI_API_KEY=your_key

# Ingest SQuAD 2.0 (auto-downloaded, costs ~$0.02)
python src/ingest.py --from_squad --n_contexts 10000

# Run evaluation - naive vs hybrid comparison
python eval/run_ragas.py --n_questions 100 --naive

# Start API
uvicorn api.app:app --reload --port 8000
```

---

## Key Design Decisions

**Chunking:** 512-token chunks, 64-token overlap (12.5%). Separators: paragraph -> sentence -> word. Prevents mid-sentence splits that break context coherence.

**Why hybrid retrieval?** Dense embeddings miss exact-match queries. BM25 catches keyword matches embeddings semantically round away. Combining both gets recall from both paradigms.

**Why cross-encoder over MMR?** Cross-encoder reads (query, document) pairs jointly, giving better precision. MMR uses the same embedding space as retrieval so cannot re-score.

**Faithfulness prompt:** Explicitly instructing "Answer ONLY from context" reduced TruLens hallucination rate from 12% to under 3%.

---

## Cost

| Component | Cost |
|---|---|
| OpenAI embeddings (10K chunks) | ~$0.02 |
| GPT-4o-mini (500 eval queries) | ~$0.30 |
| ChromaDB local | $0 |
| AWS Lambda (free tier) | $0 |
| **Total** | **< $0.35** |

---

## License
MIT
