"""Retrieval hit rate and glossary term error rate."""

import re
from pathlib import Path

import yaml
from pydantic import BaseModel

from coach.glossary import Glossary
from coach.index.search import SearchResult

EVAL_DIR = Path(__file__).parent
SLACK_S = 60  # a citation within a minute of the expected window still counts


class Expected(BaseModel):
    episode_id: str
    start: float
    end: float


class Question(BaseModel):
    question: str
    expected: list[Expected]
    verified: bool = False


class GlossaryRef(BaseModel):
    episode_id: str
    start: float
    end: float
    text: str
    verified: bool = False


def load_questions(path: Path = EVAL_DIR / "questions.yaml") -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(path.read_text(encoding="utf-8")) or []]


def load_refs(path: Path = EVAL_DIR / "glossary_refs.yaml") -> list[GlossaryRef]:
    return [GlossaryRef(**r) for r in yaml.safe_load(path.read_text(encoding="utf-8")) or []]


def citations(result: SearchResult) -> list[dict]:
    """Everything the coach would cite, in the order it would see it."""
    return [e.doc for c in result.concepts for e in c.evidence] + [h.doc for h in result.chunks]


def first_hit_rank(cited: list[dict], expected: list[Expected]) -> int | None:
    for rank, doc in enumerate(cited, start=1):
        for e in expected:
            if (
                doc["episode_id"] == e.episode_id
                and doc["start_ts"] <= e.end + SLACK_S
                and doc["end_ts"] >= e.start - SLACK_S
            ):
                return rank
    return None


def _contains(text: str, term: str) -> bool:
    # "win-con" and "win con" are the same term, not a transcription error.
    text, term = text.replace("-", " "), term.replace("-", " ")
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE) is not None


def terms_in(text: str, glossary: Glossary) -> list[str]:
    return [t for t in glossary.terms() if _contains(text, t)]


def term_errors(reference: str, hypothesis: str, glossary: Glossary) -> tuple[int, list[str]]:
    """(number of glossary terms in the reference, the ones missing from the hypothesis)."""
    expected = terms_in(reference, glossary)
    return len(expected), [t for t in expected if not _contains(hypothesis, t)]
