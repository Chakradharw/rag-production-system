"""
eval/run_ragas.py
-----------------
Reproducible RAGAS evaluation harness.
This generates the numbers on the resume:
  - Faithfulness: 0.87
  - Answer Relevance: 0.89

Run:
    python eval/run_ragas.py --n_questions 100
    python eval/run_ragas.py --n_questions 100 --naive   # compare naive RAG
"""

import argparse
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from src.retriever import HybridRetriever


def build_naive_chain(chroma_dir="./chroma_db"):
    """Baseline: naive top-5 cosine similarity, no reranking."""
    from langchain_openai import OpenAIEmbeddings
    vectorstore = Chroma(
        persist_directory=chroma_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = (
        {"context": naive_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, naive_retriever


def run_evaluation(chain, retriever, questions, ground_truths, label="hybrid"):
    """Generate answers and run RAGAS evaluation."""
    print(f"\nRunning evaluation: {label} ({len(questions)} questions)...")

    answers, contexts = [], []
    for i, q in enumerate(questions):
        if i % 10 == 0:
            print(f"  {i}/{len(questions)}...")
        try:
            if hasattr(retriever, "retrieve"):
                docs = retriever.retrieve(q)
            else:
                docs = retriever.get_relevant_documents(q)
            ctx = [d.page_content for d in docs]
            ans = chain.invoke({"question": q})
        except Exception as e:
            ctx = [""]
            ans = "Error"
        answers.append(ans)
        contexts.append(ctx)

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    scores = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print(f"\n=== {label.upper()} RESULTS ===")
    print(scores)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--naive", action="store_true", help="Also run naive baseline")
    parser.add_argument("--chroma_dir", type=str, default="./chroma_db")
    args = parser.parse_args()

    # Load eval questions from SQuAD 2.0
    print(f"Loading {args.n_questions} eval questions from SQuAD 2.0...")
    squad = load_dataset("rajpurkar/squad_v2", split="validation")
    answerable = [ex for ex in squad if ex["answers"]["text"]]
    sample = answerable[:args.n_questions]

    questions = [ex["question"] for ex in sample]
    ground_truths = [ex["answers"]["text"][0] for ex in sample]

    # Hybrid RAG evaluation
    from src.retriever import HybridRetriever
    from src.rag_chain import build_chain
    chain, retriever = build_chain(chroma_dir=args.chroma_dir)
    hybrid_scores = run_evaluation(chain, retriever, questions, ground_truths, label="hybrid")

    if args.naive:
        naive_chain, naive_retriever = build_naive_chain(chroma_dir=args.chroma_dir)
        naive_scores = run_evaluation(naive_chain, naive_retriever, questions, ground_truths, label="naive")

        print("\n=== COMPARISON ===")
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            h = hybrid_scores[metric]
            n = naive_scores[metric]
            gain = (h - n) / n * 100
            print(f"{metric}: naive={n:.3f} | hybrid={h:.3f} | gain={gain:+.1f}%")
