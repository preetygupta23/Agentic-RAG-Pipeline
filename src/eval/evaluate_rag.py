import os
import shutil
import pandas as pd
from dotenv import load_dotenv
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


def run_evaluation():
    print("🚀 Starting Agentic RAG Evaluation Pipeline...")

    # 1. CLEANUP: Delete old database to prevent schema migration crashes
    db_path = "mlflow.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🧹 Cleaned up old mlflow.db to prevent schema conflicts.")

    # 2. PATH MAPPING: Locate the dataset in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "golden_dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Could not find golden_dataset.csv at: {dataset_path}")

    print(f"📊 Loading evaluation dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 3. SIMULATION: LangGraph Agent execution loop
    print("🧠 Simulating LangGraph Agent execution...")
    generated_answers = []
    retrieved_contexts = []

    for index, row in df.iterrows():
        # This simulates your agent's response
        dummy_answer = f"Generated answer for: {row['question']}"
        dummy_context = [f"Retrieved context for: {row['question']}"]

        generated_answers.append(dummy_answer)
        retrieved_contexts.append(dummy_context)

    df['answer'] = generated_answers
    df['contexts'] = retrieved_contexts

    # 4. DATA PREP: Convert to HuggingFace Dataset
    ragas_dataset = Dataset.from_pandas(df)

    # 5. EVALUATION: Initialize Llama 3.3 70B as judge
    print("⚖️ Initializing Llama 3.3 70B LLM-as-a-Judge...")
    evaluator_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    faithfulness.llm = evaluator_llm
    answer_correctness.llm = evaluator_llm

    # 6. TRACKING: Run MLflow and Ragas
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Agentic_RAG_Evaluation")

    print("📈 Executing mathematical RAGAS evaluation scoring...")
    with mlflow.start_run() as run:
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_correctness]
        )

        result_df = result.to_pandas()

        # Log metrics
        mlflow.log_metric("mean_faithfulness", result_df['faithfulness'].mean())
        mlflow.log_metric("mean_answer_correctness", result_df['answer_correctness'].mean())

        # 7. EXPORT: Save report to project root for CI/CD artifact collection
        project_root = os.path.dirname(os.path.dirname(script_dir))
        report_output_path = os.path.join(project_root, "evaluation_report.csv")

        result_df.to_csv(report_output_path, index=False)
        mlflow.log_artifact(report_output_path)

        print(f"✅ Success! Report generated at: {report_output_path}")


if __name__ == "__main__":
    run_evaluation()