"""
Ingestion script - run this ONCE to build the vector database.

What it does (the 4 RAG ingestion steps):
  1. Read every PDF in the /pdfs folder.
  2. Split each PDF into small overlapping chunks.
  3. Tag each chunk with metadata (year, document type, page number).
  4. Convert each chunk to a vector and store it in ChromaDB.

After this finishes, your vector database lives in /chroma_db and
the RAG pipeline can search it.

USAGE:
    python -m src.ingest
"""
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from src.config import (
    PDF_FOLDER,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    DOCUMENT_METADATA,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

console = Console()


# ------------------------------------------------------------
# STEP 1: Read PDF text
# ------------------------------------------------------------
def extract_pages_from_pdf(pdf_path: Path) -> list[dict]:
    """Open a PDF and return a list of {page_num, text} dictionaries."""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = clean_text(text)
        if text.strip():
            pages.append({"page_num": page_num, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """Light cleanup - remove excessive whitespace and weird characters."""
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    text = re.sub(r"[• ]+", " ", text) # bullets, nbsp
    return text.strip()


# ------------------------------------------------------------
# STEP 2: Chunk the text
# ------------------------------------------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of ~size characters.
    Tries to break on sentence boundaries when possible.
    """
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break on a sentence boundary near the end
        slice_text = text[start:end]
        last_period = slice_text.rfind(". ")
        if last_period > size * 0.5:  # only honor it if it's not too early
            end = start + last_period + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    # Filter out tiny leftover chunks
    return [c for c in chunks if len(c) > 50]


# ------------------------------------------------------------
# STEP 3: Determine metadata for a file
# ------------------------------------------------------------
def get_metadata_for_file(filename: str) -> dict:
    """Look up year + doc_type from config, or guess from filename."""
    if filename in DOCUMENT_METADATA:
        return DOCUMENT_METADATA[filename].copy()

    # Fallback: guess from filename
    year = 2022
    for y in (2025, 2024, 2023, 2022):
        if str(y) in filename or str(y)[2:] in filename:
            year = y
            break

    doc_type = "handbook" if "andbook" in filename or "SHB" in filename else "prospectus"
    return {"year": year, "doc_type": doc_type}


# ------------------------------------------------------------
# MAIN INGEST LOOP
# ------------------------------------------------------------
def main():
    console.rule("[bold cyan]SZABIST Vision Navigator - Ingestion[/bold cyan]")

    if not PDF_FOLDER.exists():
        console.print(f"[red]ERROR:[/red] PDF folder not found: {PDF_FOLDER}")
        sys.exit(1)

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]ERROR:[/red] No PDFs found in {PDF_FOLDER}")
        sys.exit(1)

    console.print(f"[green]Found {len(pdf_files)} PDF files:[/green]")
    for p in pdf_files:
        meta = get_metadata_for_file(p.name)
        console.print(f"  - {p.name} [dim](year={meta['year']}, type={meta['doc_type']})[/dim]")

    # Set up Chroma + embedding model
    console.print(f"\n[yellow]Loading embedding model: {EMBEDDING_MODEL}...[/yellow]")
    console.print("[dim](first run downloads ~80MB - one-time)[/dim]")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # Delete and recreate collection (so reruns start clean)
    try:
        client.delete_collection(COLLECTION_NAME)
        console.print("[dim]Cleared old collection.[/dim]")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Process each PDF
    total_chunks = 0
    for pdf_path in pdf_files:
        meta = get_metadata_for_file(pdf_path.name)
        console.print(f"\n[bold]Processing:[/bold] {pdf_path.name}")

        pages = extract_pages_from_pdf(pdf_path)
        console.print(f"  Extracted {len(pages)} pages.")

        # Build chunks across all pages
        ids, documents, metadatas = [], [], []
        for page in pages:
            page_chunks = chunk_text(page["text"])
            for chunk_idx, chunk in enumerate(page_chunks):
                chunk_id = f"{pdf_path.stem}_p{page['page_num']}_c{chunk_idx}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "source": pdf_path.name,
                    "year": meta["year"],
                    "doc_type": meta["doc_type"],
                    "page": page["page_num"],
                })

        # Add to Chroma in batches (Chroma has a max-batch limit)
        BATCH = 100
        for i in tqdm(range(0, len(ids), BATCH), desc="  Embedding", leave=False):
            collection.add(
                ids=ids[i:i+BATCH],
                documents=documents[i:i+BATCH],
                metadatas=metadatas[i:i+BATCH],
            )
        console.print(f"  [green]Added {len(ids)} chunks.[/green]")
        total_chunks += len(ids)

    console.rule("[bold green]Done![/bold green]")
    console.print(f"Total chunks indexed: [bold]{total_chunks}[/bold]")
    console.print(f"Database location:    [dim]{CHROMA_DB_PATH}[/dim]")
    console.print("\nNext step: run [bold cyan]python cli.py[/bold cyan] to ask questions!")


if __name__ == "__main__":
    main()
