"""
Configuration for the SZABIST Vision Navigator RAG system.

This file contains all the settings in ONE place so you can tweak the
system without hunting through multiple files.
"""
from pathlib import Path

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_FOLDER = PROJECT_ROOT / "pdfs"
# chroma_db lives in the original zabgpt folder on the Desktop
CHROMA_DB_PATH = Path(r"C:\Users\prime\Desktop\zabgpt\chroma_db")
COLLECTION_NAME = "szabist_docs"

# ------------------------------------------------------------
# DOCUMENT METADATA MAPPING
# ------------------------------------------------------------
# This is the "trust tier" / "freshness" map. The ingest script reads
# each PDF's filename and assigns it a year + document type. The RAG
# pipeline uses this to prefer newer documents over older ones.
#
# If you add new PDFs later, add them here (or the script will guess
# from the filename).
# ------------------------------------------------------------
DOCUMENT_METADATA = {
    "Prospectus-2022.pdf":            {"year": 2022, "doc_type": "prospectus"},
    "Prospectus-2023-24.pdf":         {"year": 2023, "doc_type": "prospectus"},
    "Prospectus-2024.pdf":            {"year": 2024, "doc_type": "prospectus"},
    "Prospectus 2025-26 Final.pdf":   {"year": 2025, "doc_type": "prospectus"},
    "StudentHandbook22.pdf":          {"year": 2022, "doc_type": "handbook"},
    "Updated SHB-2023-24-1.pdf":      {"year": 2023, "doc_type": "handbook"},
    "Student Handbook-24 (23Sep).pdf":{"year": 2024, "doc_type": "handbook"},
    "Student-Handbook-2025-26.pdf":   {"year": 2025, "doc_type": "handbook"},
}

# The newest year. Anything from this year is treated as the
# "current truth" by the conflict resolver.
CURRENT_YEAR = 2025

# ------------------------------------------------------------
# CHUNKING SETTINGS
# ------------------------------------------------------------
CHUNK_SIZE = 400       # characters per chunk
CHUNK_OVERLAP = 80     # characters of overlap between chunks

# ------------------------------------------------------------
# EMBEDDING MODEL
# ------------------------------------------------------------
# Runs LOCALLY on your computer - no API key required.
# all-MiniLM-L6-v2 is small (~80MB) and fast.
# ------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ------------------------------------------------------------
# RETRIEVAL SETTINGS
# ------------------------------------------------------------
TOP_K = 10             # how many chunks to retrieve per query
RECENCY_BOOST = 0.05   # bonus score per year of recency (0.0 to disable)

# ------------------------------------------------------------
# LLM SETTINGS
# ------------------------------------------------------------
# Read from .env file at runtime, with sensible defaults
GROQ_MODEL = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1   # low temperature = more factual answers
LLM_MAX_TOKENS = 1000
