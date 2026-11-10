"""Look inside the vote: for each tune question, where did each list put the answer, and what did fusion do with it?

Per question (one rung):
  question   rank of the first answer chunk when searching with the question alone (= r0)
  rw1..rwN   its rank in each rewrite's list (top 20; - = not found)
  fused      its rank after RRF
  overlap    mean Jaccard of the rewrite lists' top-20 sets (1.0 = every rewrite found the same chunks)
  new        chunks per list that no other list found (what each list adds to the pool)
Summary: helped / hurt / same versus the question alone; in pool = some list found it (what the vote had to work
with); rescued = the question missed it, a rewrite found it; drift = the question found it, no rewrite did.

Run from rag_fusion_in_depth/ (rewrites come from the cache; a new question or style calls the LLM):
    uv run fusion/diagnose.py f1b            # per question
    uv run fusion/diagnose.py f1b f2p f2k    # one summary line per rung
"""

import statistics
import sys
from itertools import combinations

import contextual  # noqa: F401
import ladder as ctx
from eval_harness import load_golden
from retrieve import dense_search

from rag_fusion import CANDIDATES, FUSION, fuse, fusion_lists, split

INF = 10**6


def rank_in(parts, hits) -> int | None:
    return ctx.first_rank(parts, [h.text for h in hits])


def show(r) -> str:
    return f"{r:>3}" if r else "  -"


def analyse(rung: str) -> list[dict]:
    list_opts, fuse_opts = split(FUSION[rung][1])
    rows = []
    for g in load_golden(ctx.TUNE):
        if g.question_type == "no_answer":
            continue
        parts = ctx.fragments(g)
        lists = fusion_lists(g.question, **list_opts)
        sets = {name: {h.chunk_id for h in hits} for name, hits in lists.items()}
        rw_sets = [s for name, s in sets.items() if name != "question"]
        rows.append({
            "id": g.id, "own": rank_in(parts, dense_search(g.question, CANDIDATES)),
            "fused": rank_in(parts, fuse(lists, **fuse_opts)),
            "ranks": {name: rank_in(parts, hits) for name, hits in lists.items()},
            "overlap": statistics.mean(len(a & b) / len(a | b) for a, b in combinations(rw_sets, 2))
            if len(rw_sets) > 1 else 1.0,
            "new": [len(s - set().union(*(o for m, o in sets.items() if m != name))) for name, s in sets.items()],
            "pool": len(set().union(*sets.values())),
        })
    return rows


def verdicts(rows) -> dict:
    rw_found = lambda r: any(v for n, v in r["ranks"].items() if n != "question")
    return {"helped": [r["id"] for r in rows if (r["fused"] or INF) < (r["own"] or INF)],
            "hurt": [r["id"] for r in rows if (r["fused"] or INF) > (r["own"] or INF)],
            "in_pool": [r["id"] for r in rows if r["own"] or rw_found(r)],
            "rescued": [r["id"] for r in rows if not r["own"] and rw_found(r)],
            "drift": [r["id"] for r in rows if r["own"] and not rw_found(r)]}


def detail(rung: str) -> None:
    rows = analyse(rung)
    names = list(rows[0]["ranks"])
    print(f"{rung}: rank of the first answer chunk (top {CANDIDATES} per list, - = not found)\n")
    print(f"{'id':<9} question  " + " ".join(f"{n:>4}" for n in names) + "  fused  overlap  new per list   pool")
    for r in rows:
        print(f"{r['id']:<9} {show(r['own']):>8}  " + " ".join(f"{show(r['ranks'][n]):>4}" for n in names)
              + f"  {show(r['fused']):>5}  {r['overlap']:7.2f}  {str(r['new']):<14} {r['pool']:>4}")
    v = verdicts(rows)
    print(f"\nhelped {len(v['helped'])}: {', '.join(v['helped'])}\nhurt {len(v['hurt'])}: {', '.join(v['hurt'])}"
          f"\nsame {len(rows) - len(v['helped']) - len(v['hurt'])}")
    print(f"drift (question found it, no rewrite did): {', '.join(v['drift']) or 'none'}")
    print(f"rescued (question missed it, a rewrite found it): {', '.join(v['rescued']) or 'none'}")
    print(f"mean overlap {statistics.mean(r['overlap'] for r in rows):.2f}, "
          f"mean pool {statistics.mean(r['pool'] for r in rows):.0f} chunks from {len(names)} lists of {CANDIDATES}")


def summary(rungs: list[str]) -> None:
    print("| rung | overlap | pool | in pool | rescued | drift | helped | hurt |\n|---|---|---|---|---|---|---|---|")
    for rung in rungs:
        rows = analyse(rung)
        v = verdicts(rows)
        print(f"| {rung} | {statistics.mean(r['overlap'] for r in rows):.2f} "
              f"| {statistics.mean(r['pool'] for r in rows):.0f} | {len(v['in_pool'])}/{len(rows)} "
              f"| {len(v['rescued'])} | {len(v['drift'])} | {len(v['helped'])} | {len(v['hurt'])} |")


if __name__ == "__main__":
    rungs = sys.argv[1:] or ["f1a"]
    detail(rungs[0]) if len(rungs) == 1 else summary(rungs)
