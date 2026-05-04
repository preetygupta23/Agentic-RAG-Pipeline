import pandas as pd


def create_manual_dataset():
    # We are manually defining 3 test questions based on your AI Interview QnA topic
    data = {
        "question": [
            "What are the common metrics used to evaluate AI models?",
            "What is the primary role of an AI Evaluation Engineer?",
            "What is G-Eval?"
        ],
        "ground_truth": [
            "Common metrics include BLEU, ROUGE, METEOR, and G-Eval.",
            "An AI Evaluation Engineer evaluates models to ensure they are fair, accurate, and unbiased.",
            "G-Eval is an AI evaluation metric."
        ]
    }

    # Create a Pandas DataFrame and save it as a CSV
    df = pd.DataFrame(data)
    csv_path = "golden_dataset.csv"
    df.to_csv(csv_path, index=False)

    print("--- SUCCESS! ---")
    print(f"Manually verified Golden Dataset saved to {csv_path}")
    print(df.head())


if __name__ == "__main__":
    create_manual_dataset()