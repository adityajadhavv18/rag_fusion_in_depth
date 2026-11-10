"""The RAG Fusion ladder on the TUNE golden set, retrieval only. Rungs and scoring come from Contextual #2's ladder
(recall@5, MRR, nDCG@5, recall@20, MRR by category, latency, cost); results go to fusion/results/ladder_tune.json.

Run from rag_fusion_in_depth/ (Qdrant running; LLM rungs need the OpenAI key):
    set -a && source ../rag_sprints/.env && set +a && uv run fusion/fusion_ladder.py r0 f1a f1b
"""

import json
import statistics
import sys
from pathlib import Path

import contextual  # noqa: F401  (puts Contextual #2's modules on the path)
import ladder as ctx

import rewrite
from rag_fusion import FUSION, NOTE, fusion_search
from retrieve import dense_search

RESULTS = Path(__file__).resolve().parent / "results" / "ladder_tune.json"
RUNGS = ctx.RUNGS  # r0 ... r4j from Contextual #2, plus the fusion rungs
for _key, (_label, _opts) in FUSION.items():
    RUNGS[_key] = (_label, lambda q, k, o=_opts: fusion_search(q, k, **o))
# Phase 4 baseline: labels without rewrites (one dense search with the question on the note index)
RUNGS["f4q"] = ("dense with the question only, note index (no rewrites)", lambda q, k: dense_search(q, k, NOTE))


def main():
    keys = sys.argv[1:]
    if not keys:
        sys.exit(f"name the rungs to run: {', '.join(RUNGS)}")
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for key in keys:
        rewrite.MS.clear()
        results[key] = ctx.run_rung(key)
        if rewrite.MS:  # ctx reports HyDE's call time as "llm"; a fusion rung's LLM call is the rewrite
            results[key]["stage_p50_ms"]["llm"] = round(statistics.median(rewrite.MS.values()), 1)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(dict(sorted(results.items())), indent=2))
    ctx.print_table({k: results[k] for k in keys})


if __name__ == "__main__":
    main()
