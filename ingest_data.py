import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Back for large files
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS


def process_large_pdf(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found!")
        return

    print(f"📖 Loading large document: {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # NEW: We break the 20 pages into 1,000-character "bricks"
    # This makes it easier for the AI to find the EXACT paragraph later
    print("✂️ Splitting document into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100  # We overlap so no sentences are cut in half
    )
    chunks = text_splitter.split_documents(docs)

    print(f"🔢 Creating vectors for {len(chunks)} chunks...")
    embeddings = FastEmbedEmbeddings()

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("✅ Success! Your large-scale Knowledge Base is ready.")


if __name__ == "__main__":
    # UPDATE THIS to your new filename
    process_large_pdf("data/AI_Interview_QnA.pdf")