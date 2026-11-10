"""The rewriter: an LLM turns one question into several search queries (the "R" that RAG Fusion adds).

Styles are prompts; each makes the rewrites differ in a different way:
  original     Raudaschl's RAG-Fusion repo (2023), word for word: "generate multiple search queries related to"
  paraphrase   same facts, different wording and sentence shape
  keywords     the words a cricket article would print instead of the question's (last over -> final over)
  stepback     broader questions about the match / series / career the answer sits in (Zheng et al., 2023)
  decompose    one query per fact the answer needs (least-to-most, Zhou et al., 2022)
  hyde         several fake Wikipedia passages in different forms (HyDE, Gao et al., 2022, times n);
               these are embedded as DOCUMENTS, the others as queries
The cricket examples in the prompts are generic word pairs, never a golden question's match, team or player.

Two deliberate changes from the original repo: gpt-4o-mini instead of gpt-4, and temperature 0 instead of the API
default (1.0), so every rung reads the same rewrites. `temperature`/`sample` draw other samples (the stability test).

Rewrites are cached on disk per style and question (data/cache/rewrites_<style>_<model>.json) with the tokens and
time they took, like Contextual #2's HyDE passages: re-runs cost nothing, and the ladder still counts the original
call's tokens and time per question (added to hyde.USAGE, which the ladder reads and resets).
"""

import json
import re
import time
from functools import cache
from pathlib import Path

from eval_harness import settings, traced_openai

import contextual  # noqa: F401
import hyde

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
CORPUS = "a search engine over English Wikipedia cricket articles (series and match reports, tournaments, players, grounds)"
ONE_PER_LINE = "Write each query on its own line. No numbering, no introduction, nothing else.\n\nQuestion: {question}"


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


STYLES = {
    "original": [
        {"role": "system",
         "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
        {"role": "user", "content": "Generate multiple search queries related to: {question}"},
        {"role": "user", "content": "OUTPUT ({n} queries):"},
    ],
    "paraphrase": _user(f"""Rewrite the question below {{n}} different ways for {CORPUS}.
Each rewrite keeps every name, team, ground, year and number in the question but uses different words and a different
sentence shape. """ + ONE_PER_LINE),
    "keywords": _user(f"""Turn the question below into {{n}} search queries for {CORPUS}.
Use the words such an article would print, not the question's own words: cricket synonyms and full names, e.g.
- last over / final over / 20th over;  the chase / second innings;  bowled out / all out
- the third day / Day Three / third morning;  five wickets / five-wicket haul / 5/
- a ground's short name / its full name and city;  a series' nickname / its official title
Each query uses a different set of terms. Keep every name, year and number. """ + ONE_PER_LINE),
    "stepback": _user(f"""Before answering a specific cricket question, a researcher first looks up the broader
context it sits in. Write {{n}} step-back queries for {CORPUS}: broader questions about the match, innings, series,
tournament, season or career the answer belongs to (for a question about one batter's dismissal in a Test: "How did
that Test go?", "What happened in that team's second innings?"). Keep the names, years and places that identify that
context. """ + ONE_PER_LINE),
    "decompose": _user(f"""Break the question below into the separate facts needed to answer it and write one short
search query per fact for {CORPUS}, at most {{n}} queries. If it asks for a single fact, add queries for the steps
that lead to it: which match, which innings, which day. Keep every name, year and place. """ + ONE_PER_LINE),
    "hyde": _user("""You write Wikipedia articles about cricket: series and match reports, tournaments, players and
grounds. Write {n} different short passages (about 60 words each) that answer the question below, each in a different
form such an article uses: a day-by-day match report paragraph, a scorecard summary, a biography paragraph, a series or
tournament overview. The facts may be guesses; keep the question's names, years and places.
Separate the passages with a line containing only ---. No introductions.

Question: {question}"""),
}
PASSAGES = {"hyde"}  # styles whose rewrites are embedded as documents, not queries
MS: dict[str, float] = {}  # question -> rewrite call time of the last rung's call; the ladder reads and clears it


def _file(style: str) -> Path:
    return CACHE / f"rewrites_{style}_{settings.llm_model}.json"


@cache
def _cache(style: str) -> dict:
    path = _file(style)
    return json.loads(path.read_text()) if path.exists() else {}


def parse(reply: str, style: str) -> list[str]:
    """Queries: one per line, list markers and quotes stripped ("1. \"Kohli run out 2020\"" -> Kohli run out 2020).
    Passages: split on --- lines."""
    if style in PASSAGES:
        return [p.strip() for p in re.split(r"^\s*-{3,}\s*$", reply, flags=re.M) if p.strip()]
    lines = (re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip('"').strip() for line in reply.splitlines())
    return [line for line in lines if line]


def rewrites(question: str, n: int = 4, style: str = "original", temperature: float = 0.0, sample: int = 0) -> list[str]:
    """n search queries for this question (cached), adding the call's tokens and time to hyde.USAGE."""
    store = _cache(style)
    key = f"{n}|{question}" if temperature == 0 else f"{n}|t{temperature}|s{sample}|{question}"
    cached = key in store
    if not cached:
        messages = [{"role": m["role"], "content": m["content"].format(question=question, n=n)}
                    for m in STYLES[style]]
        start = time.perf_counter()
        resp = traced_openai().chat.completions.create(model=settings.llm_model, temperature=temperature,
                                                       max_tokens=900 if style in PASSAGES else 300, messages=messages)
        reply = resp.choices[0].message.content.strip()
        store[key] = {"reply": reply, "queries": parse(reply, style)[:n], "tokens_in": resp.usage.prompt_tokens,
                      "tokens_out": resp.usage.completion_tokens, "ms": (time.perf_counter() - start) * 1000}
        CACHE.mkdir(parents=True, exist_ok=True)
        _file(style).write_text(json.dumps(store, indent=1))
    hit = store[key]
    hyde.USAGE["tokens_in"] += hit["tokens_in"]
    hyde.USAGE["tokens_out"] += hit["tokens_out"]
    hyde.USAGE["llm_ms"] += hit["ms"] if cached else 0.0  # a live call is already inside the harness's timing
    MS[question] = hit["ms"]
    return hit["queries"]
