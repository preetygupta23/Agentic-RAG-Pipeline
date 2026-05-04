import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
# We use Faithfulness (Hallucination Check) and Answer Correctness (Accuracy Check)
# Add context_precision and context_recall to this line
from ragas.metrics import faithfulness, answer_correctness, context_precision, context_recall

load_dotenv()


def run_evaluation():
    print("1. Loading Dataset & Simulating LangGraph Pipeline Output...")

    data = {
        "question": [
            "What are the common metrics used to evaluate AI models?",
            "What is the primary role of an AI Evaluation Engineer?",
            "What is G-Eval?"
        ],
        "contexts": [
            ["Common metrics for AI evaluation include BLEU, ROUGE, METEOR, and G-Eval."],
            [
                "The primary role of an AI Evaluation Engineer is to evaluate models to ensure they are fair, accurate, and unbiased."],
            ["G-Eval is an AI evaluation metric."]
        ],
        "answer": [
            "AI models are typically evaluated using metrics such as BLEU, ROUGE, METEOR, and G-Eval.",
            "They evaluate AI models to make sure they are fair, accurate, and free of bias.",
            "G-Eval is a framework that uses Large Language Models to evaluate AI outputs."
        ],
        "ground_truth": [
            "Common metrics include BLEU, ROUGE, METEOR, and G-Eval.",
            "An AI Evaluation Engineer evaluates models to ensure they are fair, accurate, and unbiased.",
            "G-Eval is an AI evaluation metric."
        ]
    }

    df = pd.DataFrame(data)

    print("2. Initializing AI Judges (Groq 70B & HuggingFace)...")
    # Using the massive 70B model for flawless JSON grading
    base_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("3. Running RAGAS Spectrometer... (This takes 10-20 seconds)")
    hf_dataset = Dataset.from_pandas(df)

    # Calculate Faithfulness & Answer Correctness
    # Add the new metrics to the list
    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_correctness, context_precision, context_recall],
        llm=base_llm,
        embeddings=base_embeddings
    )

    print("\n========================================")
    print("--- PHASE 4: EVALUATION COMPLETE ---")
    print("========================================")
    print(result)

    report_path = "final_evaluation_report.csv"
    df_result = result.to_pandas()
    df_result.to_csv(report_path, index=False)
    print(f"\nDetailed metric breakdown saved to '{report_path}'")


if __name__ == "__main__":
    run_evaluation()