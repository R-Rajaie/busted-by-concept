"""Per-episode hotword list: core slang plus champions named in the title or description.

Whisper's prompt budget is about 224 tokens, so the list is capped.
"""

import re

from coach.glossary import Glossary
from coach.ingest.models import Episode

TOKEN_BUDGET = 200  # leave headroom under Whisper's ~224


def _approx_tokens(term: str) -> int:
    # Rough BPE estimate: proper nouns split into ~1.5 tokens per word, plus a separator.
    return round(len(term.split()) * 1.5) + 1


def mentioned_champions(episode: Episode, glossary: Glossary) -> list[str]:
    text = f"{episode.title}\n{episode.description}"
    found = []
    for c in glossary.champions:
        if re.search(rf"(?<!\w){re.escape(c.name)}(?!\w)", text, flags=re.IGNORECASE):
            found.append(c.name)
    return found


def build(episode: Episode, glossary: Glossary, budget: int = TOKEN_BUDGET) -> list[str]:
    # Episode-specific champions first: they're the ones most likely to be misheard here.
    candidates = mentioned_champions(episode, glossary)
    candidates += [s.term for s in glossary.slang if s.core]
    out, used = [], 0
    for term in dict.fromkeys(candidates):
        cost = _approx_tokens(term)
        if used + cost > budget:
            break
        out.append(term)
        used += cost
    return out
