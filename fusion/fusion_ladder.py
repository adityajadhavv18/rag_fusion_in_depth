"""The RAG Fusion ladder on the TUNE golden set, retrieval only. Rungs and scoring come from Contextual #2's ladder
(recall@5, MRR, nDCG@5, recall@20, MRR by category, latency, cost); results go to fusion/results/ladder_tune.json.

Run from rag_fusion_in_depth/ (Qdrant running; LLM rungs need the OpenAI key):
    set -a && source ../rag_sprints/.env && set +a && uv run fusion/fusion_ladder.py r4j
"""

import json
import sys
from pathlib import Path

import contextual  # noqa: F401  (puts Contextual #2's modules on the path)
import ladder as ctx

RESULTS = Path(__file__).resolve().parent / "results" / "ladder_tune.json"
RUNGS = ctx.RUNGS  # r0 ... r4j from Contextual #2; fusion rungs are added here


def main():
    keys = sys.argv[1:]
    if not keys:
        sys.exit(f"name the rungs to run: {', '.join(RUNGS)}")
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for key in keys:
        results[key] = ctx.run_rung(key)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(dict(sorted(results.items())), indent=2))
    ctx.print_table({k: results[k] for k in keys})


if __name__ == "__main__":
    main()
