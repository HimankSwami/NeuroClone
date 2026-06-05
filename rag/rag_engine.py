"""
rag_engine.py — Neuro's RAG System
====================================
Two collections in ChromaDB:
  • 'memory'    — auto-indexed conversation turns
  • 'knowledge' — user-dropped documents from the knowledge/ folder

Embeddings: nomic-embed-text via Ollama (fully local, no API key needed)
"""

import os
import re
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# ── Optional document loaders ──────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
EMBED_MODEL      = "nomic-embed-text"   # ollama pull nomic-embed-text
OLLAMA_URL       = "http://localhost:11434"
CHROMA_DIR       = Path("data/chroma_db")
KNOWLEDGE_DIR    = Path("knowledge")
CHUNK_SIZE       = 400    # characters per chunk
CHUNK_OVERLAP    = 80     # overlap between chunks
MEMORY_TOP_K     = 4      # how many memory hits to return
KNOWLEDGE_TOP_K  = 4      # how many knowledge hits to return


class NeuroRAG:
    """Single entry-point for all RAG operations in Neuro."""

    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        self._embed_fn = OllamaEmbeddingFunction(
            url=f"{OLLAMA_URL}/api/embeddings",
            model_name=EMBED_MODEL,
        )

        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        self._memory    = self._client.get_or_create_collection(
            name="memory",
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._knowledge = self._client.get_or_create_collection(
            name="knowledge",
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Index any new files found in knowledge/ on startup
        self.sync_knowledge_folder()
        logger.info(
            "NeuroRAG ready — memory:%d  knowledge:%d",
            self._memory.count(),
            self._knowledge.count(),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def save_memory(self, user_msg: str, assistant_msg: str) -> None:
        """Persist a conversation turn as a searchable memory."""
        turn_text = f"User: {user_msg}\nNeuro: {assistant_msg}"
        doc_id    = _uid(turn_text)
        timestamp = datetime.utcnow().isoformat()
        try:
            self._memory.upsert(
                ids=[doc_id],
                documents=[turn_text],
                metadatas=[{"timestamp": timestamp, "type": "conversation"}],
            )
            logger.debug("Memory saved: %s", doc_id)
        except Exception as e:
            logger.warning("Memory save failed (collection desynced?): %s", e)
            self._safe_count(self._memory, "memory")  # triggers recreation

    def retrieve_memory(self, query: str, top_k: int = MEMORY_TOP_K) -> list[dict]:
        """Fetch relevant past conversation turns for a query."""
        return self._query(self._memory, query, top_k)

    def retrieve_knowledge(self, query: str, top_k: int = KNOWLEDGE_TOP_K) -> list[dict]:
        """Fetch relevant knowledge-base chunks for a query."""
        return self._query(self._knowledge, query, top_k)

    def retrieve_all(self, query: str) -> list[dict]:
        """Combined retrieval from both collections (memory + knowledge)."""
        hits = self.retrieve_memory(query) + self.retrieve_knowledge(query)
        # De-duplicate by id and sort by distance (ascending = more similar)
        seen, unique = set(), []
        for h in sorted(hits, key=lambda x: x["distance"]):
            if h["id"] not in seen:
                seen.add(h["id"])
                unique.append(h)
        return unique

    def build_context_block(self, query: str) -> str:
        """
        Returns a formatted context string ready to inject into the prompt.
        Returns empty string when nothing relevant is found.
        """
        hits = self.retrieve_all(query)
        if not hits:
            return ""

        lines = ["[Neuro's Contextual Memory & Knowledge]"]
        for h in hits:
            src   = h["metadata"].get("source", h["metadata"].get("type", "memory"))
            score = round(1 - h["distance"], 3)   # cosine similarity
            lines.append(f"— [{src} | relevance {score}]: {h['document']}")
        return "\n".join(lines)

    def sync_knowledge_folder(self) -> int:
        """
        Scan the knowledge/ directory, index any new/changed files.
        Returns the number of newly indexed chunks.
        """
        new_chunks = 0
        for path in KNOWLEDGE_DIR.rglob("*"):
            if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS:
                new_chunks += self._index_file(path)
        return new_chunks

    def index_file(self, path: Path) -> int:
        """Manually index a single file. Returns chunk count added."""
        return self._index_file(Path(path))

    def delete_knowledge(self, source_name: str) -> None:
        """Remove all chunks that came from a specific file."""
        results = self._knowledge.get(where={"source": source_name})
        if results["ids"]:
            self._knowledge.delete(ids=results["ids"])
            logger.info("Deleted %d chunks from '%s'", len(results["ids"]), source_name)

    def stats(self) -> dict:
        return {
            "memory_count":    self._safe_count(self._memory, "memory"),
            "knowledge_count": self._safe_count(self._knowledge, "knowledge"),
            "knowledge_dir":   str(KNOWLEDGE_DIR.resolve()),
            "chroma_dir":      str(CHROMA_DIR.resolve()),
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _safe_count(self, collection, name: str) -> int:
        """Count with auto-recovery if ChromaDB collection UUID desyncs."""
        try:
            return collection.count()
        except Exception:
            logger.warning("Collection '%s' desynced — recreating.", name)
            try:
                new_col = self._client.get_or_create_collection(
                    name=name,
                    embedding_function=self._embed_fn,
                    metadata={"hnsw:space": "cosine"},
                )
                if name == "memory":
                    self._memory = new_col
                else:
                    self._knowledge = new_col
                return 0
            except Exception as e:
                logger.error("Failed to recreate collection '%s': %s", name, e)
                return 0

    def _query(self, collection, query: str, top_k: int) -> list[dict]:
        try:
            count = collection.count()
        except Exception:
            return []
        if count == 0:
            return []
        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
            hits = []
            for doc, meta, dist, uid in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                hits.append({"id": uid, "document": doc, "metadata": meta, "distance": dist})
            return hits
        except Exception as exc:
            logger.warning("RAG query failed: %s", exc)
            return []

    def _index_file(self, path: Path) -> int:
        """Parse → chunk → upsert. Skip if file hash unchanged."""
        source_name = path.name
        file_hash   = _file_hash(path)

        # Check if already indexed with the same hash
        existing = self._knowledge.get(
            where={"$and": [{"source": source_name}, {"file_hash": file_hash}]},
            limit=1,
        )
        if existing["ids"]:
            return 0  # nothing changed

        # Remove stale chunks from this file
        stale = self._knowledge.get(where={"source": source_name})
        if stale["ids"]:
            self._knowledge.delete(ids=stale["ids"])

        text = _extract_text(path)
        if not text.strip():
            return 0

        chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            return 0

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            ids.append(_uid(f"{source_name}:{file_hash}:{i}"))
            docs.append(chunk)
            metas.append({
                "source":    source_name,
                "file_hash": file_hash,
                "chunk_idx": i,
                "type":      "knowledge",
                "indexed_at": datetime.utcnow().isoformat(),
            })

        self._knowledge.upsert(ids=ids, documents=docs, metadatas=metas)
        logger.info("Indexed '%s' → %d chunks", source_name, len(chunks))
        return len(chunks)


# ── Helpers ────────────────────────────────────────────────────────────────

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".py", ".json", ".csv"}


def _extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            if not PDF_SUPPORT:
                logger.warning("PyMuPDF not installed — cannot read %s", path)
                return ""
            doc = fitz.open(str(path))
            return "\n".join(page.get_text() for page in doc)

        if suffix == ".docx":
            if not DOCX_SUPPORT:
                logger.warning("python-docx not installed — cannot read %s", path)
                return ""
            doc = DocxDocument(str(path))
            return "\n".join(p.text for p in doc.paragraphs)

        # Plain-text formats
        return path.read_text(encoding="utf-8", errors="ignore")

    except Exception as exc:
        logger.error("Failed to extract text from %s: %s", path, exc)
        return ""


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preserving sentence boundaries where possible."""
    text   = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start  = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Try to break at sentence end
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + size // 2:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 20]


def _uid(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()
