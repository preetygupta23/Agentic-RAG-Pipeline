import os
import pandas as pd
from dotenv import load_dotenv
import mlflow
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from langchain_groq import ChatGroq

# Load environment variables (API keys, etc.)
load_dotenv()


def run_evaluation():
    print("🚀 Starting Agentic RAG Evaluation Pipeline...")

    # 1. Setup absolute path tracking for the localized golden dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "golden_dataset.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Could not find golden_dataset.csv at: {dataset_path}")

    print(f"📊 Loading evaluation dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 2. Initialize your LangGraph pipeline response simulation loop
    # (In a real run, this passes questions to your actual LangGraph app graph)
    print("🧠 Simulating LangGraph Agent execution over evaluation dataset...")

    # Placeholder lists to simulate your actual agentic generation responses
    generated_answers = []
    retrieved_contexts = []

    for index, row in df.iterrows():
        question = row['question']

        # 💡 NOTE: This is where your actual LangGraph execution happens.
        # Example: response = langgraph_app.invoke({"question": question})
        # For evaluation pipeline setup, we simulate the agent response:
        dummy_answer = f"Generated answer corresponding to: {row['ground_truth']}"
        dummy_context = [f"Retrieved context segment validating core concepts of the RAG dissertation."]

        generated_answers.append(dummy_answer)
        retrieved_contexts.append(dummy_context)

    # Add evaluation targets to dataframe
    df['answer'] = generated_answers
    df['contexts'] = retrieved_contexts

    # 3. Convert Pandas DataFrame to HuggingFace Dataset format required by Ragas
    ragas_dataset = Dataset.from_pandas(df)

    # 4. Initialize Llama 3.3 70B via Groq as the high-fidelity judge
    print("⚖️ Initializing Llama 3.3 70B LLM-as-a-Judge via Groq...")
    evaluator_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    # Bind the evaluator LLM to the chosen metrics
    faithfulness.llm = evaluator_llm
    answer_correctness.llm = evaluator_llm

    # 5. Set up MLflow Tracking Environment
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Agentic_RAG_Evaluation")

    print("📈 Executing mathematical RAGAS evaluation scoring metrics...")
    with mlflow.start_run() as run:
        # Run evaluation scoring
        result = evaluate(
            dataset=ragas_dataset,
            metrics=[faithfulness, answer_correctness]
        )

        # Convert evaluation scores back to Pandas for export tracking
        result_df = result.to_pandas()

        # Log mean metrics summaries inside MLflow dashboard environment
        mlflow.log_metric("mean_faithfulness", result_df['faithfulness'].mean())
        mlflow.log_metric("mean_answer_correctness", result_df['answer_correctness'].mean())

        # 6. Save the final processed report directly back to workspace root for the CI/CD pipeline
        project_root = os.path.dirname(os.path.dirname(script_dir))
        report_output_path = os.path.join(project_root, "evaluation_report.csv")

        result_df.to_csv(report_output_path, index=False)
        mlflow.log_artifact(report_output_path)

        print(f"✅ Success! Report generated successfully at: {report_output_path}")


if __name__ == "__main__":
    run_evaluation()