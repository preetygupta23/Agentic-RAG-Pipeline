# --- PASTE THIS DATA IN YOUR make_pdf.py ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os


def create_manual_pdf(filename, text_content):
    # 1. Get the current folder where your project is
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    # 2. Create 'data' folder if it's missing
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Created folder: {data_dir}")

    file_path = os.path.join(data_dir, filename)

    try:
        # 3. Draw the PDF
        c = canvas.Canvas(file_path, pagesize=letter)
        t = c.beginText(50, 750)
        t.setFont("Helvetica", 12)

        # Write each line
        for line in text_content.strip().split('\n'):
            t.textLine(line.strip())

        c.drawText(t)
        c.showPage()
        c.save()
        print(f"✅ SUCCESS! File created at: {file_path}")

    except Exception as e:
        print(f"❌ FAILED to create PDF: {e}")


if __name__ == "__main__":
    content = """
    BITS PILANI M.TECH DISSERTATION - 2026
    PROJECT: Governed RAG Pipeline
    STUDENT: Preety Gupta
    START DATE: April 25, 2026
    SUBMISSION DEADLINE: August 15, 2026
    CORE TECH: LangGraph, Groq, FAISS, Arize Phoenix
    """
    create_manual_pdf("Mtech_Guidelines.pdf", content)