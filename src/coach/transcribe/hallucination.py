"""Repeated-line detector for Whisper hallucinations on silence and intro music."""

import re

from coach.transcribe.models import Segment

MIN_RUN = 3  # this many identical consecutive segments is a loop
MAX_NGRAM_SHARE = 0.5  # one repeated 3-gram making up half a long segment


def _norm(text: str) -> str:
    return re.sub(r"[^\w ]", "", text.lower()).strip()


def _looped_within(text: str) -> bool:
    words = _norm(text).split()
    if len(words) < 12:
        return False
    grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    top = max(grams.count(g) for g in set(grams))
    return top * 3 / len(words) >= MAX_NGRAM_SHARE


def flag(segments: list[Segment]) -> int:
    """Mark suspect segments in place. Returns how many were flagged."""
    i = 0
    while i < len(segments):
        j = i
        key = _norm(segments[i].text)
        while j + 1 < len(segments) and key and _norm(segments[j + 1].text) == key:
            j += 1
        if j - i + 1 >= MIN_RUN:
            for s in segments[i : j + 1]:
                s.suspect = True
        i = j + 1
    for s in segments:
        if not s.suspect and _looped_within(s.text):
            s.suspect = True
    return sum(s.suspect for s in segments)
