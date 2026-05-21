import os
from dotenv import load_dotenv # <-- Added this!

# 1. Load the API keys from your local .env file
load_dotenv() # <-- Added this!

import pandas as pd
import mlflow
from datasets import Dataset

# LangChain & Groq Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Ragas Imports (v0.1.22)
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall
)

# ==========================================
# 1. CONFIGURATION & API KEYS
# ==========================================
# Ensure the API key is loaded from the environment (GitHub Actions or local .env)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set. Please check your .env file or GitHub Secrets.")

# ==========================================
# 2. DATASET PREPARATION
# ==========================================
print("1. Loading Dataset & Simulating LangGraph Pipeline Output...")

data = {
    "question": ["What is the primary function of the Guardrail Node?"],
    "answer": ["The Guardrail Node intercepts prompts to prevent injections and off-topic queries."],
    "contexts": [[
                     "The Guardrail Node sits before retrieval. It uses low-temperature inference to block adversarial injections and off-topic requests."]],
    "ground_truth": ["It blocks adversarial prompt injections and off-topic queries before retrieval."]
}

# Convert to HuggingFace Dataset format (Required by RAGAS)
hf_dataset = Dataset.from_dict(data)

# ==========================================
# 3. INITIALIZE AI JUDGES
# ==========================================
print("2. Initializing AI Judges (Groq 70B & HuggingFace)...")

# The LLM Judge used to grade the answers (High reasoning capabilities required)
base_llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# The Embeddings model used to calculate context precision/recall
base_embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ==========================================
# 4. MLFLOW TRACKING & RAGAS EVALUATION
# ==========================================
if __name__ == "__main__":
    print("3. Starting RAGAS Evaluation with MLflow Observability...")

    # Configure MLflow to save data locally in a folder named 'mlruns'
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Agentic_RAG_Evaluation")

    # Start the tracking run
    with mlflow.start_run(run_name="Llama3_70B_Judge_Baseline"):
        # --- LOG INPUT PARAMETERS ---
        mlflow.log_param("generation_model", "llama-3.1-8b-instant")
        mlflow.log_param("judge_model", "llama-3.3-70b-versatile")
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("guardrail_temperature", 0.0)
        mlflow.log_param("ragas_version", "0.1.22")

        # --- RUN RAGAS SPECTROMETER ---
        # This calculates the metrics based on the hf_dataset provided
        result = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_correctness, context_precision, context_recall],
            llm=base_llm,
            embeddings=base_embeddings
        )

        # --- LOG OUTPUT METRICS TO MLFLOW ---
        mlflow.log_metric("avg_faithfulness", result["faithfulness"])
        mlflow.log_metric("avg_answer_correctness", result["answer_correctness"])
        mlflow.log_metric("avg_context_precision", result["context_precision"])
        mlflow.log_metric("avg_context_recall", result["context_recall"])

        # --- SAVE & LOG ARTIFACTS ---
        # Save the detailed row-by-row report locally
        df = result.to_pandas()
        csv_path = "final_evaluation_report.csv"
        df.to_csv(csv_path, index=False)

        # Upload the CSV into MLflow so it is attached to this specific run in the UI
        mlflow.log_artifact(csv_path)

        print("\n========================================")
        print("--- PHASE 4: EVALUATION COMPLETE ---")
        print("========================================")
        print(result)
        print(f"\nDetailed metrics logged to MLflow and saved to '{csv_path}'")