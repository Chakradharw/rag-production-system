"""
src/retriever.py
----------------
Hybrid retriever: dense (ChromaDB) + BM25 sparse + cross-encoder re-ranking.

This is the core differentiator over naive top-k cosine similarity:
- Dense: captures semantic similarity
- BM25: catches exact-match keywords dense embeddings miss
- Cross-encoder: reranks combined candidates for precision
"""

import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_DIR = "./chroma_db"


class HybridRetriever:
    def __init__(
        self,
        chroma_dir: str = CHROMA_DIR,
        dense_k: int = 20,
        sparse_k: int = 20,
        final_k: int = 5,
        use_reranker: bool = True,
    ):
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.final_k = final_k
        self.use_reranker = use_reranker

        # Load dense retriever
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
        )

        # Build BM25 index over all stored chunks
        print("Building BM25 index...")
        self.chunks = self.vectorstore.get()["documents"]
        tokenized = [doc.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(self.chunks)} chunks.")

        # Load cross-encoder reranker
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            print("Cross-encoder reranker loaded.")

    def retrieve(self, query: str) -> list[Document]:
        """
        1. Dense retrieval -> top dense_k candidates
        2. BM25 retrieval -> top sparse_k candidates
        3. Merge + deduplicate
        4. Cross-encoder rerank -> top final_k
        """
        # Dense
        dense_docs = self.vectorstore.similarity_search(query, k=self.dense_k)

        # BM25 sparse
        scores = self.bm25.get_scores(query.lower().split())
        top_bm25_idx = np.argsort(scores)[::-1][:self.sparse_k]
        sparse_texts = [self.chunks[i] for i in top_bm25_idx]
        sparse_docs = [Document(page_content=t) for t in sparse_texts]

        # Merge + deduplicate
        seen = set()
        merged = []
        for doc in dense_docs + sparse_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        if not merged:
            return []

        # Cross-encoder rerank
        if self.use_reranker and len(merged) > self.final_k:
            pairs = [(query, doc.page_content) for doc in merged]
            rerank_scores = self.reranker.predict(pairs)
            ranked = sorted(
                zip(rerank_scores, merged), key=lambda x: x[0], reverse=True
            )
            return [doc for _, doc in ranked[:self.final_k]]

        return merged[:self.final_k]
