"""Deterministic corrections: known mishearings from slang.yaml, then fuzzy + phonetic
matching against champion and item names.

Fuzzy matches are applied only when both spelling and sound agree strongly. Weaker
matches are returned as hints for the LLM pass instead of being applied.
"""

import re
from dataclasses import dataclass
from functools import cached_property

import jellyfish
from rapidfuzz import fuzz, process

from coach.glossary import Glossary

APPLY_RATIO = 90  # spelling similarity needed to auto-apply (with matching sound)
HINT_RATIO = 75  # weaker matches become LLM hints
MIN_TERM_LEN = 4  # short names ("Vi", "Zed") collide with ordinary words
MAX_NGRAM = 3

_WORD = re.compile(r"[A-Za-z][A-Za-z'.]*")


def _sound(text: str) -> str:
    code = jellyfish.metaphone(re.sub(r"[^A-Za-z]", "", text))
    return re.sub(r"(.)\1+", r"\1", code)  # doubled letters sound the same


def _match_case(found: str, term: str) -> str:
    """Keep a sentence-initial capital when the canonical term is lowercase."""
    return term[0].upper() + term[1:] if found[0].isupper() else term


@dataclass
class Match:
    span: str
    term: str
    ratio: float
    same_sound: bool

    @property
    def confident(self) -> bool:
        return self.same_sound and self.ratio >= APPLY_RATIO


class Matcher:
    def __init__(self, glossary: Glossary):
        self.glossary = glossary

    @cached_property
    def misheard(self) -> list[tuple[re.Pattern[str], str]]:
        pairs = [(m, s.term) for s in self.glossary.slang for m in s.misheard]
        pairs.sort(key=lambda p: -len(p[0]))  # longest first, so phrases win over words
        return [(re.compile(rf"(?<!\w){re.escape(m)}(?!\w)", re.IGNORECASE), t) for m, t in pairs]

    @cached_property
    def proper_nouns(self) -> dict[str, str]:
        """lowercased name -> canonical name, for champions and items."""
        names = [c.name for c in self.glossary.champions] + [i.name for i in self.glossary.items]
        return {n.lower(): n for n in names if len(n) >= MIN_TERM_LEN}

    def apply_slang(self, text: str) -> str:
        for pattern, term in self.misheard:
            text = pattern.sub(lambda m, t=term: _match_case(m.group(), t), text)
        return text

    def find(self, text: str) -> list[Match]:
        """Fuzzy candidates for spans that look like proper nouns but aren't spelled as one."""
        words = list(_WORD.finditer(text))
        candidates: list[tuple[int, int, Match]] = []
        for n in range(1, MAX_NGRAM + 1):
            for i in range(len(words) - n + 1):
                start, end = words[i].start(), words[i + n - 1].end()
                span = text[start:end]
                # Whisper usually capitalizes names even when it misspells them.
                if not span[0].isupper():
                    continue
                if span.lower() in self.proper_nouns:
                    # Already a correct name; it claims its span so nothing overlapping fires.
                    term = self.proper_nouns[span.lower()]
                    candidates.append((start, end, Match(span, term, 101, True)))
                    continue
                hit = process.extractOne(
                    span.lower(),
                    self.proper_nouns.keys(),
                    scorer=fuzz.ratio,
                    score_cutoff=HINT_RATIO,
                )
                if hit is None:
                    continue
                term = self.proper_nouns[hit[0]]
                match = Match(span, term, hit[1], _sound(span) == _sound(term))
                candidates.append((start, end, match))

        # Best matches first; a weak multi-word match must not block a strong single word.
        candidates.sort(key=lambda c: (c[2].confident, c[2].ratio), reverse=True)
        taken: list[tuple[int, int]] = []
        matches: list[Match] = []
        for start, end, m in candidates:
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            if m.ratio <= 100:  # skip spans that were already spelled correctly
                matches.append(m)
        return matches

    def apply_fuzzy(self, text: str) -> tuple[str, list[Match]]:
        """Apply confident matches. Returns the new text and the unapplied hints."""
        hints = []
        for m in self.find(text):
            if m.confident:
                text = re.sub(rf"(?<!\w){re.escape(m.span)}(?!\w)", m.term, text)
            else:
                hints.append(m)
        return text, hints
