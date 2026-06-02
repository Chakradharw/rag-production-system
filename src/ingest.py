"""
src/ingest.py
-------------
Document ingestion: load -> chunk -> embed -> store in ChromaDB.

Run:
    python src/ingest.py --data_dir data/corpus
    python src/ingest.py --from_squad   # use SQuAD 2.0 directly
"""

import argparse
import os
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # 12.5% overlap prevents context splits at sentence boundaries


def load_from_squad(n_contexts: int = 10_000) -> list[Document]:
    """Load unique context passages from SQuAD 2.0 as documents."""
    from datasets import load_dataset
    dataset = load_dataset("rajpurkar/squad_v2", split="train")
    contexts = list(set([ex["context"] for ex in dataset]))[:n_contexts]
    print(f"Loaded {len(contexts)} unique passages from SQuAD 2.0")
    return [Document(page_content=c, metadata={"source": "squad_v2"}) for c in contexts]


def load_from_directory(data_dir: str) -> list[Document]:
    """Load .txt files from a directory."""
    docs = []
    for path in Path(data_dir).glob("**/*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    print(f"Loaded {len(docs)} documents from {data_dir}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Chunking strategy: 512 tokens, 64-token overlap.
    Separators tried in order - prevents mid-sentence splits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def embed_and_store(chunks: list[Document], persist_dir: str = CHROMA_DIR) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print(f"Embedding {len(chunks)} chunks with text-embedding-3-small...")

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.persist()
    print(f"Stored in ChromaDB at {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--from_squad", action="store_true")
    parser.add_argument("--n_contexts", type=int, default=10_000)
    args = parser.parse_args()

    if args.from_squad:
        docs = load_from_squad(n_contexts=args.n_contexts)
    elif args.data_dir:
        docs = load_from_directory(args.data_dir)
    else:
        print("Use --from_squad or --data_dir. Defaulting to SQuAD 2.0.")
        docs = load_from_squad()

    chunks = chunk_documents(docs)
    embed_and_store(chunks)
    print("Ingestion complete.")
