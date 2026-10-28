"""LLM proper-noun pass. It may only fix names and jargon; any chunk whose text changes by
more than MAX_CHANGE is rejected and keeps its pre-LLM text.
"""

import json

from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein

from coach import llm
from coach.correct.matcher import Match
from coach.glossary import Glossary

MAX_CHANGE = 0.05
CHUNK_SEGMENTS = 40

SYSTEM = """You fix transcription errors in a League of Legends coaching podcast transcript.

Only fix misspelled or misheard proper nouns and game jargon: champion names, ability names,
item names, objectives, summoner spells, and League slang. Do not fix grammar, filler words,
punctuation, false starts, or style. Do not rephrase anything. If you are not confident a
word is a misheard term, leave it alone.

Return only the segments you changed, with their full corrected text."""


class _Fixed(BaseModel):
    id: int
    text: str


class _Fixes(BaseModel):
    segments: list[_Fixed]


def glossary_context(glossary: Glossary, hotwords: list[str]) -> str:
    slang = "\n".join(f"- {s.term}: {s.meaning}" for s in glossary.slang)
    champs = ", ".join(c.name for c in glossary.champions)
    items = ", ".join(i.name for i in glossary.items)
    return (
        f"Episode keywords: {', '.join(hotwords)}\n\nSlang:\n{slang}\n\n"
        f"Champions: {champs}\n\nItems: {items}"
    )


def change_ratio(before: list[str], after: list[str]) -> float:
    total = sum(len(t) for t in before) or 1
    return sum(Levenshtein.distance(b, a) for b, a in zip(before, after, strict=True)) / total


def correct_chunk(
    segments: list[tuple[int, str]], hints: list[Match], context: str
) -> dict[int, str]:
    """Return `{segment_id: corrected_text}` for the segments the LLM changed."""
    payload = [{"id": i, "text": t} for i, t in segments]
    hint_text = "\n".join(f'- "{h.span}" might be "{h.term}"' for h in hints) or "(none)"
    prompt = (
        f"{context}\n\nPossible mishearings found by a fuzzy matcher (verify, don't trust):\n"
        f"{hint_text}\n\nTranscript segments:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    fixes = llm.complete_model(prompt, SYSTEM, _Fixes, model=llm.FAST_MODEL)
    ids = {i for i, _ in segments}
    return {f.id: f.text.strip() for f in fixes.segments if f.id in ids}
