"""Answers for a fusion rung: Contextual #2's answer step (answer.py) unchanged, so the comparison with r4j is fair.

Same answer prompt (cricket reading rules, [n] citations, the exact "I don't know" line), same top 5 under their
match labels, same judges: RAGAS faithfulness / relevancy, abstention, and answer.py's correctness grader against the
golden reference. Only the retrieval differs. Results go to fusion/results/answers_<split>.json.

f6a is f5a's search with one sentence added to the prompt's partnership rule. On tune, f5a's model read "how many did
Dhoni and Gambhir add together?" as a sum (91 + 97 = 188) although two of its five passages said "added 109": the rule
covered the passage side ("a partnership is the figure the text gives") but not the question's wording.

Run from rag_fusion_in_depth/ (Qdrant running, OpenAI key sourced):
    uv run fusion/fusion_answer.py f5a f6a            # tune split + judge
    uv run fusion/fusion_answer.py r0 r4j f5a --test   # the final held-out run
"""

from pathlib import Path

import contextual  # noqa: F401
import answer
import fusion_ladder  # noqa: F401  (adds the fusion rungs to Contextual #2's RUNGS, which answer.py reads)

answer.RESULTS = Path(__file__).resolve().parent / "results"

PARTNERSHIP = """- A partnership or stand ("added", "put on", "shared") is the figure the text gives for it. Never add up the two
  batters' own scores: they include runs made with other partners."""
assert PARTNERSHIP in answer.PROMPT
PROMPTS = {"original": answer.PROMPT,
           "v2": answer.PROMPT.replace(PARTNERSHIP, """- A partnership or stand ("added", "put on", "shared") is the figure the text gives for it. A question asking
  how many two batters added, added together, put on or shared asks for that partnership figure. Never add up the
  two batters' own scores: they include runs made with other partners.""")}
PROMPT_OF = {"f6a": "v2"}  # rung -> answer prompt; every other rung keeps Contextual #2's

answer.RUNGS["f6a"] = ("f5a, answer prompt v2 (partnership questions)", answer.RUNGS["f5a"][1])
_make_pipeline = answer.make_pipeline


def make_pipeline(rung: str):
    answer.PROMPT = PROMPTS[PROMPT_OF.get(rung, "original")]  # read by the pipeline at call time
    return _make_pipeline(rung)


answer.make_pipeline = make_pipeline

if __name__ == "__main__":
    answer.main()
