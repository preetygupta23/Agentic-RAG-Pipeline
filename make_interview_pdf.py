from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
import os


def create_interview_pdf(filename):
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    path = os.path.join(data_dir, filename)
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter

    # --- Title Page ---
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 100, "AI Evaluation & QA Engineer")
    c.setFont("Helvetica", 16)
    c.drawCentredString(width / 2, height - 130, "Technical Interview Master Question Bank")
    c.line(100, height - 150, 500, height - 150)

    # --- Content ---
    y_position = height - 200

    questions = [
        ("SECTION 1: LLM EVALUATION METRICS", [
            "1. Explain the difference between BLEU, ROUGE, and METEOR scores.",
            "2. Why is G-Eval (LLM-as-a-judge) becoming more popular than traditional metrics?",
            "3. How do you measure 'Hallucination Rate' in a production RAG system?",
            "4. Define 'Faithfulness' vs 'Answer Relevance' in the RAGAS framework."
        ]),
        ("SECTION 2: RAG & RETRIEVAL TESTING", [
            "1. How do you test the 'Context Recall' of a vector database like FAISS?",
            "2. What is 'Lost in the Middle' phenomenon in long-context LLMs?",
            "3. Describe a test case for a multi-hop retrieval query.",
            "4. How do you evaluate the impact of different chunking strategies on accuracy?"
        ]),
        ("SECTION 3: AI GOVERNANCE & RED TEAMING", [
            "1. What is a 'Prompt Injection' attack and how do you write a test for it?",
            "2. How do you evaluate an AI for 'Bias and Fairness' in recruiting use cases?",
            "3. Describe the process of 'Jailbreaking' an LLM for safety testing.",
            "4. What are the key components of an AI Incident Response plan?"
        ]),
        ("SECTION 4: MLOPS & MONITORING", [
            "1. How does Arize Phoenix help in tracing 'Root Cause' for poor LLM answers?",
            "2. What is 'Data Drift' and how is it different from 'Concept Drift' in AI?",
            "3. How do you automate regression testing for a stochastic (random) LLM output?"
        ])
    ]

    for section, q_list in questions:
        if y_position < 100:  # New Page if running out of space
            c.showPage()
            y_position = height - 50

        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.dodgerblue)
        c.drawString(50, y_position, section)
        y_position -= 25

        c.setFont("Helvetica", 11)
        c.setFillColor(colors.black)
        for q in q_list:
            c.drawString(70, y_position, q)
            y_position -= 20
        y_position -= 15

    c.save()
    print(f"✅ Created interview PDF: {path}")


if __name__ == "__main__":
    create_interview_pdf("AI_Interview_QnA.pdf")