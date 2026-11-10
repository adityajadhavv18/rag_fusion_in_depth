# RAG Fusion #3: cricket on Wikipedia

Ask "who took five wickets when England were bowled out for 67 in the Headingley Test of the 2019 Ashes?" and get "Josh Hazlewood, 5/30 [1]". Contextual RAG #2 answered "Jofra Archer" here: his five wickets came in the other innings. RAG Fusion asks the question several ways before it searches, and lets the answers vote.

```
QUERY:  question ── gpt-4o-mini writes 4 keyword rewrites ("five-wicket haul Headingley 2019 Ashes England all out 67")
           │
           ├─ 5 queries (the question + 4 rewrites), each searched 3 ways on Contextual #2's labelled index:
           │     dense (bge-small) · BM25 · SPLADE   → 15 ranked lists, top 20 each
           │
           ├─ Reciprocal Rank Fusion, k=1, every list equal → top 20
           ├─ gpt-4o-mini reads the 20 under their match labels, ranks them → top 5
           └─ gpt-4o-mini answers with [n] citations, or "I don't know"
```

Think of a jury. One juror reading the question one way can miss the page that answers it, because the page says "final over" and the question says "last over". Five jurors, each phrasing it differently, each hand in a ranked list, and the pages that many of them rank high win. The whole build is about two things a jury can get wrong: **who sits on it** (how the rewrites differ, what they search) and **how the votes are counted**.

- **RAG Fusion** (Raudaschl, 2023; evaluated by Rackauckas, 2024): LLM rewrites → one search each → RRF. Phase 1 builds it exactly as published.
- **Reciprocal Rank Fusion** (Cormack et al., 2009), **CombSUM / CombMNZ** (Fox & Shaw, 1994), round-robin: four ways to count the votes.
- Rewrite styles: plain **paraphrase**, cricket **keywords**, **step-back** (Zheng et al., 2023), **decomposition** (least-to-most, Zhou et al., 2022), several **HyDE** passages (Gao et al., 2022).
- **Listwise LLM reranking** (RankGPT, Sun et al., 2023) after the vote, unchanged from Contextual #2.

## Same corpus, same index, same ruler as Contextual #2

This build reuses [contextual-rag-in-depth](https://github.com/adityajadhavv18/contextual-rag-in-depth) as-is: the 140 cricket Wikipedia articles, the 4,650 section-aware chunks, the four Qdrant collections (bare chunk, title › headings, + match box, + LLM note), its retrievers, HyDE, the LLM reranker, the ladder scoring and the answer step. **Nothing was re-indexed**, so every number here is directly comparable with #2. `fusion/contextual.py` puts its `cricket/` folder on the import path.

The golden set is the same: **30 tune / 20 held-out test** questions in `eval_harness`, by category:

- **which**: which match? ("how was Root out in England's second innings of the first Test of the 2023 Ashes?")
- **twin**: the same players in another match (Kohli in Adelaide **2020**, not 2014 or 2018)
- **pron**: the answer paragraph says "the visitors" or "he", never the name
- **num**: scorecard figures
- **multi**: two facts in one question
- **bio**: player and ground articles
- **none**: no answer exists; the right reply is "I don't know"

Every setting was chosen on tune. The test split was run once, at the end.

## How it works

**1. Rewrites** (`fusion/rewrite.py`). One gpt-4o-mini call writes 4 queries. Six prompt styles were measured; the winner, **keywords**, asks for the words a cricket article would print instead of the question's own:

```
Use the words such an article would print, not the question's own words: cricket synonyms and full names, e.g.
- last over / final over / 20th over;  the chase / second innings;  bowled out / all out
- the third day / Day Three / third morning;  five wickets / five-wicket haul / 5/
- a ground's short name / its full name and city;  a series' nickname / its official title
```

The examples are generic word pairs, never a golden question's match, team or player. Rewrites are cached per style and question with their tokens and time, so re-runs cost nothing and still count the original call.

**2. Fifteen lists** (`fusion/rag_fusion.py`). The question and each rewrite search dense, BM25 and SPLADE on `cricket_header_match_note`: the chunk under its article › headings, match box and LLM note.

**3. The vote**. RRF with **k=1**: a list's #1 gets 1/2 of a point, its #19 gets 1/20. The published setting, k=60, gives #1 only 1.3× the points of #19, so a chunk that several lists rank lukewarmly beats the one a single list is sure of.

**4. Rerank and answer**. Contextual #2's listwise reranker reads the fused top 20 and keeps 5; its answer prompt (cricket reading rules, `[n]` citations, the exact refusal line) writes the answer, with one sentence added (Phase 6, below).

```python
queries = [question] + gpt_4o_mini(KEYWORDS_PROMPT, question)[:4]              # 1 call, cached
lists   = {f"{s}_{i}": search(s, q, index="header_match_note", top=20)
           for i, q in enumerate(queries) for s in ("dense", "bm25", "splade")}   # 15 lists
top20   = rrf_fuse(lists, k=1)[:20]
top5    = llm_rerank(question, top20)[:5]
```

`fusion/diagnose.py` opens the vote for any rung: where each list put the answer, how much the lists overlap, what each rewrite adds that no other list found, and which questions fusion helped, hurt, rescued or lost to drift.

## Run it

From `rag_fusion_in_depth/`, next to a working `contextual_rag_in_depth/` (its articles downloaded and its four collections built). Qdrant runs in Docker (`open -a Docker`, then `docker start qdrant`). The OpenAI key is needed for rewrites, the reranker and answers.

```bash
uv run fusion/fusion_ladder.py r0 f1a f1b            # retrieval-only rungs on tune (any rung name below)
uv run fusion/diagnose.py f1a                        # inside the vote, per question
uv run fusion/diagnose.py f1b f2p f2k f2s f2d f2h    # one summary line per rung
uv run fusion/fusion_answer.py f5a f6a               # answers + judge on tune
uv run fusion/fusion_answer.py f1a2 r4j2 f6a --test  # the final held-out run
```

## Results

### The ladder (tune, 28 answerable questions, retrieval only)

Each phase changed one thing. One question moves MRR by about 0.04, and three samples of the same setting at temperature 0.7 scored 0.29–0.32, so read gaps under ~0.04 as noise. p50 includes the LLM calls' original time.

| Rung | What changed | recall@5 | MRR | recall@20 | p50 | Kept? |
|---|---|---|---|---|---|---|
| r0 | dense only, bare index (Contextual's baseline) | 0.41 | 0.26 | 0.50 | 12 ms | baseline |
| f1a | **RAG Fusion as published**: 4 rewrites, dense, RRF k=60 | 0.46 | 0.32 | 0.52 | 2.0 s | ✅ |
| f1b | + the question as a 5th list | 0.46 | 0.31 | 0.52 | 2.0 s | ✅ safety net |
| f2p / f2s / f2d / f2h | rewrite style: paraphrase / step-back / decompose / HyDE ×4 | 0.29–0.43 | 0.23–0.27 | 0.43–0.61 | 1.7–6.3 s | ❌ |
| **f2k** | **rewrite style: cricket keywords** | 0.43 | 0.32 | **0.64** | 1.9 s | ✅ |
| f2k1–f2k8 | 1, 2, 3, 5, 8 rewrites (f2k has 4) | 0.39–0.46 | 0.25–0.34 | 0.61–0.68 | 1.4–3.1 s | 4 kept |
| **f3k1** | **RRF k=1** (k=5, 10, 20: 0.34) | **0.50** | **0.35** | 0.64 | 1.9 s | ✅ |
| f3sum / f3mnz / f3rr | CombSUM / CombMNZ / round-robin | 0.43–0.46 | 0.29–0.30 | 0.61–0.68 | 1.9 s | ❌ |
| f3k1w2 / w3 | the question's lists count ×2 / ×3 | 0.50 | 0.32 / 0.31 | 0.64 | 1.9 s | ❌ |
| f4q | *labels, no rewrites: the question alone, dense, labelled index* | 0.68 | 0.46 | 0.95 | 13 ms | reference |
| f4d | f3k1 on the labelled index | 0.73 | 0.55 | 0.88 | 2.0 s | ✅ |
| f4s1 | + BM25 and SPLADE with the question | 0.80 | 0.59 | 0.95 | 2.0 s | ✅ |
| **f4s2** | **+ BM25 and SPLADE with every rewrite** | **0.88** | **0.69** | 0.91 | 2.1 s | ✅ |
| f4s2k20 | f4s2 with Contextual's k=20 | 0.84 | 0.62 | 0.95 | 2.1 s | ❌ |
| r3a | *Contextual #2: HyDE + BM25 + SPLADE, labelled index, no rerank* | 0.84 | 0.66 | 0.95 | 3.2 s | reference |
| **f5a** | **f4s2 + LLM rerank of the top 20** | **0.91** | **0.91** | 0.91 | **3.8 s** | ✅ **winner** |
| f5h | f4s2 + Contextual's HyDE list, no rerank | 0.84 | 0.69 | 0.95 | 5.2 s | ❌ |
| f5b | f5h + LLM rerank | 0.91 | 0.91 | 0.95 | 6.7 s | ❌ |
| r4j | *Contextual #2's best: HyDE hybrid + LLM rerank* | 0.91 | 0.89 | 0.95 | 4.5 s | the bar |

MRR by category:

| | which | twin | pron | num | multi | bio |
|---|---|---|---|---|---|---|
| r0 | 0.17 | 0.40 | 0.00 | 0.33 | 0.38 | 0.40 |
| f1a (published) | 0.00 | 0.40 | 0.25 | 0.42 | 0.62 | 0.36 |
| f3k1 | 0.12 | 0.40 | 0.07 | 0.57 | 0.42 | 0.73 |
| f4s2 | 0.51 | 0.60 | 0.90 | 0.57 | 0.83 | 0.83 |
| **f5a** | **0.89** | 0.60 | **1.00** | **1.00** | **1.00** | **1.00** |
| r4j | 0.75 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |

On tune, f5a's answers were **0.92** correct, the same as r4j. One sentence added to the answer prompt (f6a, below) made it **0.93**.

### The final run (held-out test, 18 answerable + 2 no-answer, judged)

The three new runs share one answer prompt (v2), so they differ only in their search. r0 is Contextual #2's saved run with its original prompt.

| | r0 (dense only) | f1a2 (published RAG Fusion) | r4j2 (Contextual's best) | **f6a (this build)** |
|---|---|---|---|---|
| **answers correct** | 0.60 | 0.70 | 0.93 | **0.98** |
| evidence in top 5 (recall@5) | 0.50 | 0.61 | 1.00 | **1.00** |
| MRR | 0.32 | 0.53 | 0.91 | **0.92** |
| faithfulness | 0.81 | 0.73 | 0.89 | **0.94** |
| answer relevancy | 0.68 | 0.77 | 0.94 | 0.94 |
| context precision | 0.43 | 0.60 | 0.93 | **0.98** |
| context recall | 0.52 | 0.58 | 0.94 | 0.94 |
| answerable questions refused | 28% | 17% | 0% | **0%** |
| no-answer questions refused | 100% | 100% | 100% | 100% |
| cost / question | $0.00025 | $0.00034 | $0.00127 | $0.00125 |

**f6a and r4j are tied.** The 0.05 gap is two grader calls, not two better answers:
- **multi-06** (who ran out Guptill in the 2019 final): all three stacks gave the same facts, "Buttler; England 26 boundaries to 17". The grader marked it correct for f6a and for r4j's original run, partial for r4j2. Strictly, all three are partial: the reference says Jason Roy *and* Buttler.
- **twin-08** (Root's first century in Australia, 138\*): both stacks found the right passage and gave a wrong score. f6a said "100" (graded partial), r4j2 said "104" (graded wrong).

With 20 questions one answer is worth 0.05. Read f6a and r4j as one band, about 19 of 20. f6a is 1.00 in every category except **twin** (0.83).

"Correct" is Contextual #2's grade: the judge (gpt-4o-mini) compares each answer with the golden reference, correct 1, partial 0.5, wrong 0, and a refused answerable question scores 0.

The whole build cost about **$0.60** in API calls: rewrites for six styles, five counts and three samples (~$0.04), new reranks (~$0.06), answers and judging on tune (~$0.20) and on test (~$0.20). Nothing was embedded or indexed.

## What went wrong, and why

- **The published vote rewards lukewarm agreement (f1a).** In which-05 (Sundar and Thakur's stand at the Gabba) the question and rewrite 1 both ranked the answer **#1**, and it fused to **#13**. The *England in India* article lead sat at #19, #18 and #19 in three lists: 3/79 = 0.038 points against the answer's 1/61 + 1/76 = 0.030. With k=60, rank barely matters, so a chunk many lists half-like wins. **which** fell from 0.17 to 0.00.
- **The rewrites shared one blind spot (f1a).** The original prompt ("generate multiple search queries related to…") wrote four near-copies: keyword bags naming the tour. They agreed on the same generic chunks; the Bumrah biography came top in which-05 because every rewrite named the tour he played in.
- **More varied rewrites fill the pool but not the top (f2).** Step-back, decomposition and HyDE ×4 overlapped least (Jaccard ~0.2, ~60 distinct chunks against paraphrase's 0.57 and 33). Decomposition and HyDE ×4 put the answer in the top 20 more often (recall@20 0.59 and 0.61 against 0.52), but varied lists rarely agree, so the vote couldn't lift it to the top 5. Step-back and decomposition also drifted: they find the right match's article lead or scorecard, not the paragraph, and step-back guessed "second innings" when England's 67 was their first.
- **The number of rewrites didn't matter (f2k1–f2k8).** MRR 0.32, 0.25, 0.30, 0.32, 0.27, 0.34 for 1, 2, 3, 4, 5, 8: no trend. The pool grows from 30 to 58 chunks, but questions with the answer anywhere in it stay at about 21 of 28. One good rewrite plus the question gets most of it.
- **Adding up scores just counts heads (f3sum / f3mnz).** CombSUM and CombMNZ gave identical rankings because bge-small's cosines are squeezed: the median #1 scores 0.831, the median #20 0.784. One more list finding a chunk adds ~0.8, seventeen times the whole #1-to-#20 gap, so score fusion ignores rank. RRF's ranks are the better signal here even though all lists share one scale.
- **Weighting the user's question made it worse (f3k1w2/w3).** The keyword rewrites search better than the raw question, so extra weight mostly lets the question's mistakes win.
- **Vote counting has a ceiling (Phase 3).** Every counting method landed between 0.28 and 0.35. In 7 of 28 questions no list found the answer at all: the paragraphs never name their own match, and rewording the question can't put that name back. That is Contextual #2's anchor loss, and only the labelled index fixes it.
- **The reranker can only rank what it's given (f5a).** twin-04 (Rahane's 112 at the MCG) fused to #21, one place outside the 20 the reranker reads, and was lost; r4j had it at #3.
- **The answer model read "add together" as a sum (Phase 6).** which-06's top 5 held "Gambhir and Dhoni **added 109**" twice, and the model still answered 91 + 97 = 188. The prompt's partnership rule covered the passages ("a partnership is the figure the text gives") but not the question's wording. Prompt v2 adds: *"A question asking how many two batters added, added together, put on or shared asks for that partnership figure."* which went to 1.00, nothing else moved.
- **It still guesses when evidence is missing.** twin-03 (2017 IPL final, last over): no list finds the Johnson chunk, and the model answers "Bumrah" from memory. twin-04 and test's twin-08 take a mid-innings score (104\*, 100) for the final one.
- **The grader is noisy.** Identical answers got different faithfulness scores between runs (which-03: 0.33, then 0.00), and the correctness judge graded the same Guptill answer correct twice and partial once. On 20–30 questions, read every gap under ~0.05 as a tie.

## What it gets right

- **Keyword rewrites give exact-word search the article's words.** The main finding. Dense search already knows "last over" means "final over"; BM25 and SPLADE only match words that are there. Running every rewrite through them took tune MRR from 0.59 to 0.69 and **pron from 0.48 to 0.90** (in pron-02 all five SPLADE lists ranked the answer #1; the dense lists had it at #5–6 or not at all).
- **Same quality as Contextual's best, without HyDE.** Tune MRR 0.91 against r4j's 0.89, test 0.98 against 0.93: one band. Four short keyword queries take 1.9 s to write against HyDE's 3.2 s for a 100-word passage, so retrieval p50 on tune was 3.8 s against 4.5 s. (End to end on test, with every call live, f6a measured 6.2 s p50: API latency varies run to run.)
- **Fixed what Contextual got wrong on pron-01.** r4j answered Archer, whose five wickets came in the other innings; f5a answers Hazlewood 5/30.
- **Honest refusals.** Both no-answer questions refused on tune and test; no answerable question refused.
- **Cheap to try.** No re-indexing. Fusion's own cost is $0.00007 a question; the reranker is ~90% of the bill.
- **Every step is measured.** One change per rung, the same chunks and golden set as Contextual #2, and a diagnostics tool that shows *why* a rung moved.

## Ideas not built

- **Two more answer rules**, to be tuned on tune: "when a dismissal lists several fielders (run out (Roy/Buttler)), name them all", and "an innings' score is where it ended, not a score at stumps or at a milestone" (twin-04, twin-08).
- **A steadier grader**: grade each answer three times and take the majority.
- **A reranker pool of 25.** twin-04 sat at #21. A deeper pool didn't help Contextual's twin-03, so it may cost more than it saves.
- **One mixed-style call**: a paraphrase, two keyword queries and one step-back in the same reply, so variety comes without drift.
- **Run the rewrite and HyDE calls in parallel** if HyDE is kept: f5b's 6.7 s would drop to about 4.5 s. It added no quality here.
- **Refuse when the excerpts don't name the asked match** (Corrective RAG): the remaining guesses would become "I don't know".

## When I'd reach for it

When users ask in **different words from the documents**: "last over" vs "final over", "five-for" vs "five-wicket haul", a team's nickname vs its full name. Put the rewrites into **keyword search** (BM25 / SPLADE), which can't bridge synonyms on its own; dense search mostly can. Write **keyword-style rewrites**, not paraphrases or step-back questions: paraphrases add nothing new, step-back drifts. Use **RRF with a small k** (1–20) so a confident hit isn't outvoted, and don't expect vote counting to fix what no list finds. Fusion widens the search; it can't give a chunk context it lacks, so if paragraphs don't name their subject, label the chunks first (Contextual RAG). With an LLM reranker on top, fusion's job is simply to get the answer into the reranker's 20.

## Files

| File | Purpose |
|---|---|
| `fusion/contextual.py` | Puts `contextual_rag_in_depth/cricket` on the import path; no module here may reuse its module names |
| `fusion/rewrite.py` | The six rewrite prompts, the parser, the cache per style and question |
| `fusion/rag_fusion.py` | The lists (dense / sparse, any index, + HyDE), the four vote counts, the rerank step, and `FUSION`: every rung's options |
| `fusion/fusion_ladder.py` | Runs rungs with Contextual #2's scoring (recall@5, MRR, nDCG@5, recall@20, by category, latency, cost) |
| `fusion/diagnose.py` | Inside the vote: per-list ranks, overlap, new chunks per list, helped / hurt / rescued / drift |
| `fusion/fusion_answer.py` | Contextual #2's answer step on fusion rungs; prompt v2 and the test-run aliases f1a2 / r4j2 |
| `fusion/results/ladder_tune.json` | Every rung's tune numbers |
| `fusion/results/answers_tune.json`, `answers_test.json` | Judged answers, by category and per question |
