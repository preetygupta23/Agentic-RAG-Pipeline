import streamlit as st
import pandas as pd
import os


def main():
    # Set up the web page layout
    st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

    st.title("🎯 Agentic RAG Pipeline: Final Evaluation Dashboard")
    st.markdown("""
    This dashboard provides a quantitative analysis of the LangGraph-orchestrated Retrieval-Augmented Generation pipeline. 
    Metrics are calculated using a 70-Billion parameter LLM Judge via the RAGAS framework.
    """)
    st.divider()

    # --- SMART PATH FINDER ---
    # This checks multiple folders to find exactly where PyCharm saved your CSV
    possible_paths = [
        "final_evaluation_report.csv",
        "src/eval/final_evaluation_report.csv",
        os.path.join(os.path.dirname(__file__), "final_evaluation_report.csv"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "final_evaluation_report.csv")
    ]

    df = None
    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.toast(f"Report loaded successfully from: {path}")  # Tiny pop-up to show it worked
            break

    if df is None:
        st.error(
            "⚠️ Evaluation report not found! Please ensure 'evaluate_rag.py' ran successfully and generated the CSV.")
        return

    # --- SECTION 1: AVERAGE METRICS ---
    st.header("Overall System Performance")

    # Calculate averages
    avg_faithfulness = df['faithfulness'].mean()
    avg_correctness = df['answer_correctness'].mean()
    avg_precision = df['context_precision'].mean()
    avg_recall = df['context_recall'].mean()

    # Display large KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Faithfulness (Hallucination Check)", value=f"{avg_faithfulness:.2f}")
    col2.metric(label="Answer Correctness", value=f"{avg_correctness:.2f}")
    col3.metric(label="Context Precision", value=f"{avg_precision:.2f}")
    col4.metric(label="Context Recall", value=f"{avg_recall:.2f}")

    st.divider()

    # --- SECTION 2: VISUALIZATION CHARTS ---
    st.header("Question-by-Question Analysis")
    st.markdown("Visualizing metric variance across the evaluation dataset.")

    # Select only the numeric metric columns for the chart
    chart_data = df[['faithfulness', 'answer_correctness', 'context_precision', 'context_recall']]
    st.bar_chart(chart_data)

    st.divider()

    # --- SECTION 3: RAW DATA AUDIT ---
    st.header("Evaluation Audit Log")
    st.markdown("Review the raw generation outputs and ground truth pairs.")
    # Display the dataframe cleanly
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()