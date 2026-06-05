import os
import pandas as pd
from dotenv import load_dotenv
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper  # <--- IMPORTANT BRIDGE
from langchain_groq import ChatGroq

load_dotenv()


def run_evaluation():
    print("🚀 Starting Agentic RAG Evaluation Pipeline...")

    # 1. Cleanup old DB
    db_path = "mlflow.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # 2. Path mapping
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "golden_dataset.csv")
    df = pd.read_csv(dataset_path)

    # 3. Simulate LangGraph
    generated_answers = [f"Answer for {q}" for q in df['question']]
    retrieved_contexts = [["Context A", "Context B"] for _ in df['question']]
    df['answer'] = generated_answers
    df['contexts'] = retrieved_contexts
    ragas_dataset = Dataset.from_pandas(df)

    # 4. BRIDGE: Wrap the Groq LLM for Ragas compatibility
    print("⚖️ Initializing Llama 3.3 70B via Ragas Wrapper...")
    groq_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    ragas_llm = LangchainLLMWrapper(groq_llm)  # This adds set_run_config

    faithfulness.llm = ragas_llm
    answer_correctness.llm = ragas_llm

    # 5. MLflow & Evaluate
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Agentic_RAG_Evaluation")

    with mlflow.start_run():
        result = evaluate(dataset=ragas_dataset, metrics=[faithfulness, answer_correctness])
        result_df = result.to_pandas()

        mlflow.log_metric("mean_faithfulness", result_df['faithfulness'].mean())
        mlflow.log_metric("mean_answer_correctness", result_df['answer_correctness'].mean())

        project_root = os.path.dirname(os.path.dirname(script_dir))
        report_path = os.path.join(project_root, "evaluation_report.csv")
        result_df.to_csv(report_path, index=False)
        print(f"✅ Success! Report generated at: {report_path}")


if __name__ == "__main__":
    run_evaluation()