"""
src/rag_chain.py
----------------
LangChain RAG chain with faithfulness-first prompt.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from src.retriever import HybridRetriever


FAITHFULNESS_PROMPT = PromptTemplate.from_template("""
You are a precise assistant. Answer the question using ONLY the information
in the context below. If the answer is not present in the context, respond
exactly with: "I don't have enough information to answer this."

Do NOT add information from outside the context.
Do NOT speculate or make inferences beyond what is stated.

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(chroma_dir: str = "./chroma_db"):
    retriever = HybridRetriever(chroma_dir=chroma_dir)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {
            "context": lambda x: format_docs(retriever.retrieve(x["question"])),
            "question": RunnablePassthrough(),
        }
        | FAITHFULNESS_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(chain, question: str) -> dict:
    answer = chain.invoke({"question": question})
    return {"question": question, "answer": answer}
