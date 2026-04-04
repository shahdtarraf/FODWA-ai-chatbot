"""
Data Ingestion Script — LOCAL USE ONLY.
Reads PDF → splits into chunks → generates embeddings → saves FAISS index + chunks.json.

Usage:
    python app/ingest_data.py

Requirements (install locally):
    pip install -r requirements.txt

NEVER run this in production / on Render.
"""

import os
import sys
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

# Load .env
load_dotenv()

# ─── Configuration ──────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "text-embedding-3-small"

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
INDEX_PATH = os.path.join(OUTPUT_DIR, "index.faiss")
CHUNKS_PATH = os.path.join(OUTPUT_DIR, "chunks.json")


def load_all_pdfs(data_dir: str) -> tuple[str, int]:
    """Extract and merge all text from all PDFs in a directory."""
    import pdfplumber
    import glob
    import unicodedata

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {data_dir}")
        sys.exit(1)

    print(f"📄 Found {len(pdf_files)} PDF(s) in {data_dir}")
    
    def clean_arabic_text(text: str) -> str:
        # 1. Reverse each line because RTL text was extracted LTR visually
        lines = text.split('\n')
        reversed_text = '\n'.join([line[::-1] for line in lines])
        # 2. Normalize presentation forms to standard Arabic codepoints
        return unicodedata.normalize('NFKC', reversed_text)

    combined_text = ""
    for pdf_path in pdf_files:
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        combined_text += clean_arabic_text(page_text) + "\n"
        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")

    print(f"✅ Total text length extracted: {len(combined_text)} characters")
    return combined_text, len(pdf_files)


def split_text(text: str) -> list[str]:
    """Split text into chunks using LangChain text splitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"✂️  Splitting text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", "،", " ", ""]
    )

    chunks = splitter.split_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def generate_embeddings(chunks: list[str], api_key: str) -> list[list[float]]:
    """Generate embeddings for all chunks using OpenAI."""
    print(f"🧠 Generating embeddings for {len(chunks)} chunks...")

    if api_key in ["<PUT_YOUR_KEY_HERE>", "mock", "YOUR_OPENAI_API_KEY", ""]:
        print("⚠️ Mocking embeddings because a dummy API key is used.")
        return [np.random.rand(1536).tolist() for _ in chunks]

    client = OpenAI(api_key=api_key, timeout=30.0)
    embeddings = []

    # Process in batches of 20 to avoid rate limits
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"  ✅ Batch {i // batch_size + 1}: {len(batch)} chunks embedded")
        except Exception as e:
            print(f"  ❌ Batch {i // batch_size + 1} failed: {e}")
            sys.exit(1)

    print(f"✅ Total embeddings: {len(embeddings)}")
    return embeddings


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    """Build a FAISS IndexFlatL2 from embeddings."""
    print("🔨 Building FAISS index...")

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    vectors = np.array(embeddings, dtype=np.float32)
    index.add(vectors)

    print(f"✅ FAISS index built: {index.ntotal} vectors, dimension={dimension}")
    return index


def save_data(index: faiss.IndexFlatL2, chunks: list[str]):
    """Save FAISS index and chunks to disk."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 FAISS index saved: {INDEX_PATH}")

    # Save chunks as JSON
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Chunks saved: {CHUNKS_PATH}")

    # Print file sizes
    index_size = os.path.getsize(INDEX_PATH) / 1024
    chunks_size = os.path.getsize(CHUNKS_PATH) / 1024
    print(f"📊 Index size: {index_size:.1f} KB")
    print(f"📊 Chunks size: {chunks_size:.1f} KB")


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("🚀 FODWA AI — Data Ingestion Pipeline")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set. Create a .env file with your key.")
        sys.exit(1)

    # Step 1: Load all PDFs in data/
    text, pdf_count = load_all_pdfs(DATA_DIR)

    if not text.strip():
        print("❌ No text extracted from PDFs")
        sys.exit(1)

    # Step 2: Split into chunks
    chunks = split_text(text)

    if not chunks:
        print("❌ No chunks created")
        sys.exit(1)

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(chunks, api_key)

    # Step 4: Build FAISS index
    index = build_faiss_index(embeddings)

    # Step 5: Save to disk
    save_data(index, chunks)

    print("=" * 60)
    print("✅ Ingestion complete!")
    print(f"   📁 Index: {INDEX_PATH}")
    print(f"   📁 Chunks: {CHUNKS_PATH}")
    print(f"   📊 Total chunks: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
