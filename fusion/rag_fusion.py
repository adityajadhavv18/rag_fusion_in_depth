"""RAG Fusion (Raudaschl, 2023): rewrite the question several ways, search with each, merge the lists with RRF.

    question -> LLM -> q1 q2 q3 q4 -> dense search each (top 20) -> RRF (k=60) -> top 5

A chunk found high by several rewrites beats one found #1 by a single rewrite, so the merge rewards chunks that
many phrasings agree on. The original repo searches ONLY the rewrites; keep_original adds the user's own question as
one more list (a safety net for when every rewrite drifts away from it).

fusion_lists returns the separate lists (for diagnose.py); fuse merges them; fusion_search does both (for the ladder).
FUSION names every fusion rung's options, so the ladder and diagnose.py run exactly the same configuration.

How the votes are counted (fuse's `method`, Phase 3):
  rrf     weight / (k + rank), summed (Cormack et al., 2009). Small k: a #1 counts far more than a #10;
          large k: rank barely matters, so being found by MANY lists wins.      k=60: #1 = 1/61, #19 = 1/79
  sum     CombSUM: the chunk's cosine scores added up over the lists that found it (Fox & Shaw, 1994)
  mnz     CombMNZ: CombSUM x the number of lists that found it (rewards agreement even more)
  rr      round-robin: #1 of every list (the question's first), then every #2, ... (no voting at all)
All lists here are dense searches with the same model, so their cosine scores share one scale and sum/mnz are fair;
across retrievers (BM25 scores vs cosines) they would not be, which is why RRF uses ranks only.
"""

from collections import defaultdict

import contextual  # noqa: F401
from ingest import BM25, SPLADE, collection as index
from retrieve import PLAIN, Hit, _dense, dense_search, rrf_fuse, sparse_search, timed

from rewrite import PASSAGES, rewrites

CANDIDATES = 20  # how deep each list goes into the vote, as in Contextual #2's ladder
PAPER_K = 60  # RRF k in the original repo (Cormack et al.'s default)
NOTE = index("header_match_note")  # Contextual #2's best index: title › headings + match box + LLM note + chunk


def is_question(name: str) -> bool:
    """Lists searched with the user's own question: "question" (dense), "bm25_question", "splade_question"."""
    return name.endswith("question")


def fusion_lists(q: str, n: int = 4, keep_original: bool = False, collection: str = PLAIN, style: str = "original",
                 temperature: float = 0.0, sample: int = 0, sparse: str = "") -> dict[str, list[Hit]]:
    """Dense lists: rw1..rwN (+ "question"). sparse adds BM25 and SPLADE lists (Phase 4):
    "question" = searched with the question only (as in Contextual #2), "all" = with every rewrite too."""
    queries = rewrites(q, n, style, temperature, sample)
    lists = _dense_lists(q, queries, keep_original, collection, style)
    if sparse:
        texts = {"question": q} | ({f"rw{i}": x for i, x in enumerate(queries, start=1)} if sparse == "all" else {})
        for using in (BM25, SPLADE):
            lists |= {f"{using}_{name}": sparse_search(t, CANDIDATES, using, collection) for name, t in texts.items()}
    return lists


def _dense_lists(q, queries, keep_original, collection, style) -> dict[str, list[Hit]]:
    if style in PASSAGES:  # fake passages are embedded as documents, like Contextual #2's HyDE
        with timed("hyde_embed"):
            vectors = [v.tolist() for v in _dense().passage_embed(queries)]
        lists = {f"rw{i}": dense_search(x, CANDIDATES, collection, vector=v)
                 for i, (x, v) in enumerate(zip(queries, vectors), start=1)}
    else:
        lists = {f"rw{i}": dense_search(x, CANDIDATES, collection) for i, x in enumerate(queries, start=1)}
    if keep_original:
        lists["question"] = dense_search(q, CANDIDATES, collection)
    return lists


def fuse(lists: dict[str, list[Hit]], method: str = "rrf", rrf_k: int = PAPER_K,
         weights: dict[str, float] | None = None) -> list[Hit]:
    if method == "rrf":
        return rrf_fuse(lists, k=rrf_k, weights=weights)
    if method == "rr":
        order = sorted(lists, key=lambda name: not is_question(name))  # the question's lists go first
        seen = {}
        for rank in range(CANDIDATES):
            for name in order:
                if rank < len(lists[name]):
                    seen.setdefault(lists[name][rank].chunk_id, lists[name][rank])
        return list(seen.values())
    with timed("fuse"):  # sum / mnz
        total, count, by_id = defaultdict(float), defaultdict(int), {}
        for name, hits in lists.items():
            for h in hits:
                total[h.chunk_id] += (weights or {}).get(name, 1.0) * h.score
                count[h.chunk_id] += 1
                by_id.setdefault(h.chunk_id, h)
        points = {c: total[c] * (count[c] if method == "mnz" else 1) for c in total}
        return [Hit(by_id[c].chunk_id, by_id[c].doc_id, by_id[c].path, by_id[c].text, points[c])
                for c in sorted(points, key=lambda c: (-points[c], c))]


FUSE_OPTS = ("method", "rrf_k", "weights")


def split(opts: dict) -> tuple[dict, dict]:
    """A rung's options -> (fusion_lists options, fuse options)."""
    return ({k: v for k, v in opts.items() if k not in FUSE_OPTS}, {k: v for k, v in opts.items() if k in FUSE_OPTS})


def fusion_search(q: str, k: int, **opts) -> list[Hit]:
    list_opts, fuse_opts = split(opts)
    return fuse(fusion_lists(q, **list_opts), **fuse_opts)[:k]


# rung -> (label, fusion_lists options). Phase 2 keeps f1b's setup (rewrites + the question, dense, plain index,
# RRF k=60) and changes one thing per rung: the rewrite style, then the number of rewrites, then sampling.
FUSION = {
    "f1a": ("RAG Fusion as published: 4 rewrites, dense, RRF k=60", {}),
    "f1b": ("f1a + the question itself as a 5th list", {"keep_original": True}),
}
for _key, _style in {"p": "paraphrase", "k": "keywords", "s": "stepback", "d": "decompose", "h": "hyde"}.items():
    FUSION[f"f2{_key}"] = (f"f1b, rewrite style: {_style}", {"keep_original": True, "style": _style})
for _n in (1, 2, 3, 5, 8):  # f2k has 4
    FUSION[f"f2k{_n}"] = (f"f2k with {_n} rewrites", {"keep_original": True, "style": "keywords", "n": _n})
for _s in (1, 2, 3):  # the same prompt sampled at temperature 0.7: how much does the score move run to run?
    FUSION[f"f2kt{_s}"] = (f"f2k at temperature 0.7, sample {_s}",
                           {"keep_original": True, "style": "keywords", "temperature": 0.7, "sample": _s})

# Phase 3: f2k's lists (keywords x4 + the question, from the cache), counted differently. No LLM calls.
F2K = {"keep_original": True, "style": "keywords"}
for _k in (1, 5, 10, 20):  # f2k is k=60
    FUSION[f"f3k{_k}"] = (f"f2k, RRF k={_k}", {**F2K, "rrf_k": _k})
for _m, _name in {"sum": "CombSUM", "mnz": "CombMNZ", "rr": "round-robin"}.items():
    FUSION[f"f3{_m}"] = (f"f2k, fused by {_name}", {**F2K, "method": _m})
for _k in (1, 60):  # the user's own question counts 2x / 3x a rewrite
    for _w in (2, 3):
        FUSION[f"f3k{_k}w{_w}"] = (f"f2k, RRF k={_k}, question x{_w}", {**F2K, "rrf_k": _k, "weights": {"question": _w}})

# Phase 4: f3k1 (keywords x4 + the question, RRF k=1), changing WHERE the rewrites search.
F3K1 = {**F2K, "rrf_k": 1}
FUSION["f4d"] = ("f3k1 on the note index (dense only)", {**F3K1, "collection": NOTE})
FUSION["f4s1"] = ("f4d + BM25 and SPLADE with the question", {**F3K1, "collection": NOTE, "sparse": "question"})
FUSION["f4s2"] = ("f4d + BM25 and SPLADE with the question and every rewrite",
                  {**F3K1, "collection": NOTE, "sparse": "all"})
FUSION["f4s2k20"] = ("f4s2 with RRF k=20 (Contextual #2's k)", {**F3K1, "collection": NOTE, "sparse": "all", "rrf_k": 20})
