"""Reuse Contextual RAG #2 (../contextual_rag_in_depth) as-is: same chunks, same Qdrant collections, same scoring.

Its modules import each other by bare name (`import hyde`, `from chunks import ...`), so its cricket/ folder goes on
sys.path. Our own modules must not reuse those names (chunks, ingest, retrieve, hyde, llm_rerank, notes, ladder,
answer), or an import there would pick up ours instead. Its data paths are built from its own __file__, so the
corpus text and the HyDE / LLM-rerank caches are read (and extended) in contextual_rag_in_depth/data/cricket/.
"""

import sys
from pathlib import Path

CONTEXTUAL = Path(__file__).resolve().parents[2] / "contextual_rag_in_depth"
sys.path.append(str(CONTEXTUAL / "cricket"))
