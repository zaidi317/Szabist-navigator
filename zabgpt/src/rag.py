"""
RAG pipeline - the brain of the SZABIST Vision Navigator.

Given a question, this module:
  1. Searches the vector DB for the most relevant chunks.
  2. Boosts newer chunks (so 2025 wins over 2022 when both match).
  3. Detects when retrieved chunks DISAGREE with each other (conflict).
  4. Builds a "conflict-aware" prompt that tells the LLM exactly how
     to handle disagreements (always prefer the most recent year).
  5. Sends everything to the LLM and returns the answer + sources.

USAGE (from Python):
    from src.rag import VisionNavigator
    nav = VisionNavigator()
    result = nav.ask("Where is the Robotics Lab?")
    print(result["answer"])
"""
import os
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from src.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    RECENCY_BOOST,
    CURRENT_YEAR,
    GROQ_MODEL,
    OPENAI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

load_dotenv()


# ============================================================
# THE SYSTEM PROMPT - this is the "personality" + rules of
# the AI. It's where we encode all the conflict-aware tricks.
# ============================================================
SYSTEM_PROMPT = """You are the SZABIST Vision Navigator - an expert AI assistant for students at \
Shaheed Zulfikar Ali Bhutto Institute of Science and Technology (SZABIST).

Your job is to answer questions about SZABIST campus life, academic policies, \
office locations, course requirements, lab rules, and admissions using ONLY \
the document excerpts provided to you in each query.

## STRICT RULES YOU MUST FOLLOW:

1. **GROUND IN SOURCES**: Answer ONLY using the [SOURCE] excerpts provided. If the \
answer is not in the sources, say "I don't have information about that in the \
SZABIST documents I have access to."

2. **TEMPORAL PRIORITY (most important rule)**: Each source is tagged with a year. \
When sources from different years disagree, ALWAYS prefer the MOST RECENT year. \
The current academic year is 2025-26.

3. **FLAG CONFLICTS TRANSPARENTLY**: If you notice that older and newer sources \
contradict each other on a fact (e.g., credit hours, office locations, fees), \
mention this briefly: "Note: This was updated in 2025 - older handbooks listed it as X."

4. **CITE YOUR SOURCES**: At the end of your answer, list which sources you used \
in this format: [Source: <document_name>, year <year>, page <page>]

5. **BE CONCISE AND DIRECT**: Students need quick answers. Get to the point in 2-4 \
short paragraphs. Don't pad with phrases like "Based on the documents..." just answer.

6. **HANDLE NAVIGATION SPECIALLY**: For "where is X?" questions, give the building \
number AND room number if available (e.g., "Building 100, Room 104").

7. **NEVER MAKE UP**: If you're unsure or the sources are silent, say so. Do not \
invent course codes, room numbers, fees, or rules from your general knowledge.

8. **NEVER MIX YEARS SILENTLY**: If you use facts from a 2022 source, say so. Don't \
present old data as if it's current.
"""


# ============================================================
# Main RAG class
# ============================================================
class VisionNavigator:
    def __init__(self):
        self._setup_db()
        self._setup_llm()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    def _setup_db(self):
        if not CHROMA_DB_PATH.exists():
            raise FileNotFoundError(
                f"Vector DB not found at {CHROMA_DB_PATH}. "
                "Run `python -m src.ingest` first."
            )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=embed_fn
        )

    def _setup_llm(self):
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        self.provider = provider

        if provider == "groq":
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key or api_key == "your_groq_key_here":
                raise ValueError(
                    "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
                    "and put it in your .env file."
                )
            self.client = Groq(api_key=api_key)
            self.model = GROQ_MODEL
        elif provider == "openai":
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_key_here":
                raise ValueError("OPENAI_API_KEY not set in .env file.")
            self.client = OpenAI(api_key=api_key)
            self.model = OPENAI_MODEL
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use 'groq' or 'openai'.")

    # --------------------------------------------------------
    # STEP 1+2: Retrieval with recency boosting
    # --------------------------------------------------------
    _STOPWORDS = {
        "who", "what", "where", "when", "how", "why", "is", "are", "was",
        "the", "a", "an", "of", "in", "at", "to", "for", "and", "or",
        "about", "tell", "me", "give", "find", "does", "did", "has",
    }

    def _score_chunk(self, doc: str, meta: dict, dist: float) -> dict:
        sim_score = 1.0 - dist
        year = meta.get("year", 2022)
        recency_bonus = -(CURRENT_YEAR - year) * RECENCY_BOOST
        return {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "year": year,
            "doc_type": meta.get("doc_type", "unknown"),
            "page": meta.get("page", 0),
            "sim_score": round(sim_score, 3),
            "final_score": round(sim_score + recency_bonus, 3),
        }

    def retrieve(self, query: str, year_filter: int | None = None, top_k: int = TOP_K) -> list[dict]:
        """
        Hybrid search: vector similarity + keyword match for proper nouns.
        Pulls 3x candidates from vector search, merges keyword hits, re-ranks by recency.
        """
        where_filter = {"year": year_filter} if year_filter else None

        # --- Vector search ---
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 3,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        seen_ids = set()
        chunks = []
        for doc, meta, dist, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            seen_ids.add(chunk_id)
            chunks.append(self._score_chunk(doc, meta, dist))

        # --- Keyword search for terms likely to be proper nouns ---
        keywords = [
            w for w in query.split()
            if len(w) >= 4 and w.lower().rstrip("?.,") not in self._STOPWORDS
        ]
        for kw in keywords[:4]:
            try:
                kw_res = self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    where=where_filter,
                    where_document={"$contains": kw},
                    include=["documents", "metadatas", "distances"],
                )
                for doc, meta, dist, chunk_id in zip(
                    kw_res["documents"][0],
                    kw_res["metadatas"][0],
                    kw_res["distances"][0],
                    kw_res["ids"][0],
                ):
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        chunks.append(self._score_chunk(doc, meta, dist))
            except Exception:
                pass

        chunks.sort(key=lambda c: c["final_score"], reverse=True)
        return chunks[:top_k]

    # --------------------------------------------------------
    # STEP 3: Detect conflicts between retrieved chunks
    # --------------------------------------------------------
    def detect_conflicts(self, chunks: list[dict]) -> dict:
        """
        Group retrieved chunks by year. If we have chunks from
        multiple years, flag a potential conflict.
        """
        by_year = defaultdict(list)
        for c in chunks:
            by_year[c["year"]].append(c)

        years_present = sorted(by_year.keys(), reverse=True)
        has_conflict = len(years_present) > 1

        return {
            "has_conflict": has_conflict,
            "years_present": years_present,
            "newest_year": years_present[0] if years_present else None,
            "by_year": dict(by_year),
        }

    # --------------------------------------------------------
    # STEP 4: Build the conflict-aware prompt
    # --------------------------------------------------------
    def build_prompt(self, query: str, chunks: list[dict], conflict_info: dict) -> str:
        """Assemble the final user message sent to the LLM."""
        # Format each retrieved chunk with explicit metadata
        sources_text = ""
        for i, c in enumerate(chunks, 1):
            sources_text += (
                f"\n[SOURCE {i}] (file={c['source']} | year={c['year']} | "
                f"type={c['doc_type']} | page={c['page']})\n"
                f"{c['text']}\n"
            )

        conflict_warning = ""
        if conflict_info["has_conflict"]:
            years = ", ".join(str(y) for y in conflict_info["years_present"])
            conflict_warning = (
                f"\n** CONFLICT WARNING **\n"
                f"The sources above come from MULTIPLE academic years ({years}). "
                f"If they disagree on any fact, you MUST prefer the most recent year "
                f"({conflict_info['newest_year']}). Briefly mention the change to the user.\n"
            )

        prompt = f"""STUDENT QUESTION:
{query}

RETRIEVED SOURCES (sorted by relevance + recency):
{sources_text}
{conflict_warning}
Now answer the student's question using ONLY the sources above. \
Follow ALL the rules in your system prompt - especially the temporal priority rule \
and the citation requirement."""

        return prompt

    # --------------------------------------------------------
    # STEP 5: Call the LLM
    # --------------------------------------------------------
    def call_llm(self, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content

    # --------------------------------------------------------
    # MAIN entry point
    # --------------------------------------------------------
    def ask(self, query: str, year_filter: int | None = None) -> dict:
        """End-to-end: question in → answer + sources out."""
        chunks = self.retrieve(query, year_filter=year_filter)
        conflict_info = self.detect_conflicts(chunks)
        prompt = self.build_prompt(query, chunks, conflict_info)
        answer = self.call_llm(prompt)

        return {
            "query": query,
            "answer": answer,
            "sources": chunks,
            "conflict_info": {
                "has_conflict": conflict_info["has_conflict"],
                "years_present": conflict_info["years_present"],
                "newest_year": conflict_info["newest_year"],
            },
        }
