"""
open_fusion.prompts - judge + synthesis prompt templates and response labeling.

`label_responses` assigns labels POSITIONALLY (panel[0]->MODEL A, panel[1]->MODEL B,
...) and emits only the survivors. The letter is tied to the slot, not to the
surviving set, so 'MODEL C' always means the third panel model even if the second
failed. That stability is what makes the judge's per-model attribution auditable.
"""
from __future__ import annotations

from typing import Iterable


def label_responses(responses: Iterable) -> str:
    blocks: list[str] = []
    for i, r in enumerate(responses):
        letter = chr(ord("A") + i)
        ok = getattr(r, "ok", getattr(r, "status", "") == "ok")
        if not ok:
            continue
        content = (getattr(r, "content", "") or "").strip()
        blocks.append(f"[MODEL {letter}] (slug: {getattr(r, 'model', '?')})\n{content}")
    return "\n\n".join(blocks)


JUDGE_SYSTEM = """You are the JUDGE in a multi-model deliberation pipeline.

Several models independently answered the same question. Their answers are given to \
you labeled [MODEL A], [MODEL B], .... Your job is NOT to answer the question. Your \
job is to distill the panel into a precise, structured analysis that a later writer \
will turn into the final answer.

Output ONLY a single JSON object (no prose, no markdown fences) with exactly these keys:

{
  "consensus":        [ "claim every/most models agree on", ... ],
  "contradictions":   [ { "topic": "...",
                          "stances": [ { "model": "MODEL A", "stance": "..." },
                                       { "model": "MODEL B", "stance": "..." } ] }, ... ],
  "partial_coverage": [ { "models": ["MODEL A"], "point": "raised by some, not all" }, ... ],
  "unique_insights":  [ { "model": "MODEL B", "insight": "..." }, ... ],
  "blind_spots":      [ "important angle NO model addressed", ... ],
  "best_model":       "MODEL A" | null,
  "best_reason":      "why this model's answer is the strongest overall (completeness, correctness, reasoning quality). null if no clear winner."
}

Rules:
- Every stance and every unique insight MUST name the originating model label. This \
attribution is mandatory.
- Be specific and faithful: do not invent agreement that isn't there, and surface \
real contradictions rather than smoothing them over.
- **best_model**: Identify the single model whose answer is closest to correct and \
complete. This is used to short-circuit synthesis when one model clearly dominates. \
Set to null only if models are equally good/bad.
- If a section has no content, use an empty array [] (do not omit the key).
- Output valid JSON and nothing else."""


def JUDGE_USER(question: str, labeled: str) -> str:
    return (f"QUESTION:\n{question}\n\n"
            f"PANEL RESPONSES:\n{labeled}\n\n"
            "Produce the JSON analysis described in your instructions now.")


def JSON_RETRY(previous: str, error: str) -> str:
    return ("Your previous output was not valid according to the required schema.\n"
            f"Parser error: {error}\n\n"
            "Return ONLY the corrected JSON object — no prose, no markdown fences, no "
            "explanation. Ensure every contradiction stance and every unique insight "
            "includes a \"model\" field, and that all seven top-level keys are present "
            "(consensus, contradictions, partial_coverage, unique_insights, blind_spots, "
            "best_model, best_reason; use [] for empty arrays, null for no best model).")


SYNTHESIS_SYSTEM = """You are the SYNTHESIZER. You are given the original question, a \
structured analysis of a panel of models (consensus, contradictions with per-model \
stances, partial coverage, unique insights, blind spots), AND the full original panel \
responses.

Write the single best final answer to the question:
- Lead with what the panel agrees on where it's well-founded.
- Where models contradict, adjudicate explicitly: state which view is better \
supported and why, rather than hedging.
- Fold in the strongest unique insights from the analysis, and preserve key details \
from the original responses (e.g. full proofs, code, step-by-step reasoning) rather \
than summarizing them away.
- If one model's response is clearly superior for a section, use that section directly \
with minor integration — do NOT rewrite from scratch if the original is already excellent.
- Explicitly note material blind spots so the reader knows the limits of the answer.
- You have NO tools and NO new evidence here; the evidence is frozen. Do not invent \
facts beyond the analysis and responses.
Write directly and concisely for the end user. Do not mention "the panel", "MODEL A", \
or this process unless the question is about the deliberation itself."""


def SYNTHESIS_USER(question: str, analysis_json: str, labeled_responses: str = "") -> str:
    parts = [f"QUESTION:\n{question}\n\nSTRUCTURED ANALYSIS (JSON):\n{analysis_json}"]
    if labeled_responses:
        parts.append(f"\n\nORIGINAL PANEL RESPONSES:\n{labeled_responses}")
    parts.append("\n\nWrite the final answer now.")
    return "".join(parts)


FALLBACK_SYSTEM = """You are the SYNTHESIZER operating in FALLBACK mode: the structured \
judge step failed, so you are given the raw panel answers directly (labeled [MODEL A], \
...). Cross-check them yourself, resolve disagreements explicitly, and write the single \
best final answer to the question. You have no tools; rely only on the answers shown. \
Do not mention the labels or this process unless the question is about it."""


def FALLBACK_USER(question: str, labeled: str) -> str:
    return (f"QUESTION:\n{question}\n\nPANEL RESPONSES:\n{labeled}\n\n"
            "Write the final answer now.")
