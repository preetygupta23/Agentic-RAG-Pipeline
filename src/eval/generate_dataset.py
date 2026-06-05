import os
import pandas as pd


def generate_golden_dataset():
    print("Writing 20-question robust evaluation dataset...")

    data = {
        "question": [
            "What is the core architecture discussed in this dissertation?",
            "Which framework is used to orchestrate the agentic workflow?",
            "Which model is utilized as the LLM-as-a-Judge for evaluation?",
            "Which model is used as the primary generator in the RAG system?",
            "What embedding model is used for document vectorization?",
            "Which vector database is implemented for semantic retrieval?",
            "What primary framework is used to mathematically calculate hallucination metrics?",
            "Which Ragas metric evaluates if the generated answer is derived entirely from the retrieved context?",
            "Which Ragas metric measures the accuracy of the generated answer against the ground truth?",
            "What tool is integrated for experiment tracking and observability?",
            "What backend database format was configured for MLflow tracking in the CI/CD pipeline?",
            "What platform is used to automate the evaluation pipeline on every code push?",
            "Which Python library is used to read and manipulate the evaluation dataset?",
            "What format must the data be converted to before passing it into Ragas?",
            "Why was the Groq API selected for model inference?",
            "Which Python library is utilized for programmatic API endpoint validation?",
            "How does the LangGraph agent differ from a standard RAG chatbot?",
            "How are the final evaluation reports preserved after a GitHub Actions run completes?",
            "How are stakeholders notified of a successful pipeline run?",
            "What is the primary language used for the entire Agentic RAG and evaluation stack?"
        ],
        "ground_truth": [
            "An automated evaluation pipeline for an Agentic RAG system.",
            "LangGraph and LangChain.",
            "Llama 3.3 70B via the Groq API.",
            "Llama 3.1 8B.",
            "HuggingFace all-MiniLM-L6-v2.",
            "FAISS (Facebook AI Similarity Search).",
            "RAGAS (Retrieval Augmented Generation Assessment).",
            "Faithfulness.",
            "Answer Correctness.",
            "MLflow.",
            "A local SQLite database.",
            "GitHub Actions.",
            "Pandas.",
            "HuggingFace Dataset format.",
            "To ensure ultra-low latency token generation.",
            "The Python requests library.",
            "It uses stateful, controllable agent workflows rather than unpredictable linear generation.",
            "Using GitHub Artifacts.",
            "An automated email containing the CSV report is sent via a GitHub Actions SMTP step.",
            "Python."
        ]
    }

    df = pd.DataFrame(data)

    # Get the exact directory where this script is saved (src/eval)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Direct the output file to save in this exact folder
    output_path = os.path.join(script_dir, "golden_dataset.csv")

    df.to_csv(output_path, index=False)

    print(f"✅ Success! Golden dataset saved inside src/eval/: {output_path}")
    print(f"📊 Total rows written: {len(df)}")


if __name__ == "__main__":
    generate_golden_dataset()