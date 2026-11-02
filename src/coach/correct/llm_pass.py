"""LLM proper-noun pass. It may only fix names and jargon. A changed segment is kept only if
each edit is a few words long and brings in nothing but known terms (`segment_ok`); a chunk
whose kept text still changes by more than MAX_CHANGE is rejected whole.
"""

import json
import re
from difflib import SequenceMatcher

from pydantic import BaseModel
from rapidfuzz.distance import Levenshtein

from coach import llm
from coach.correct.matcher import Match
from coach.glossary import Glossary

MAX_CHANGE = 0.05
CHUNK_SEGMENTS = 40
MAX_SPAN = 3  # words one edit may replace; more means the model moved or rewrote text

# Words the show itself uses that no glossary list covers.
SHOW_WORDS = (
    "Broken by Concept BBC League of Legends Riot Curtis Nathan Charlie "
    "EU EUW EUNE NA OCE KR LAN LAS BR TR JP Korea Korean SoloQ "
    "hardstuck fog jungle laning ult ultimate "
    "teleport flash smite ignite exhaust heal barrier cleanse ghost "
    # champion classes, which shows say as adjectives too ("tanky", "bruisery")
    "tank bruiser juggernaut diver fighter skirmisher assassin mage battlemage burst artillery "
    "marksman enchanter catcher warden vanguard support specialist poke engage disengage peel"
)
PREFIXES = {"out", "re", "pre", "non", "anti", "over", "under"}  # "out-jungling"
# Endings that turn any known word (champion, item, slang, class) into a form of it: "tanky",
# "tankier", "jungling", "ganked", "Barons", "gankable", "aggressively".
# Each pairs an ending with what may have been dropped from the base word ("jungle" + "ing").
SUFFIXES = [
    ("s", ""), ("es", ""), ("y", ""), ("y", "e"), ("ier", "y"), ("iest", "y"), ("ier", ""),
    ("er", ""), ("er", "e"), ("ers", ""), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"),
    ("'s", ""), ("able", ""), ("able", "e"), ("ish", ""), ("ly", ""), ("ily", "y"),
    ("'d", ""), ("'ed", ""), ("'ing", ""), ("d", ""),  # "TP'd", "TP'ing", "TPd"
]  # fmt: skip

SYSTEM = """You fix transcription errors in a League of Legends coaching podcast transcript.
The podcast is "Broken by Concept" (BBC), hosted by Curtis, Nathan, and Charlie.

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


def _token(word: str) -> str:
    return re.sub(r"^[^\w'’]+|[^\w'’]+$", "", word.lower()).replace("’", "'")


def vocabulary(glossary: Glossary, hotwords: list[str]) -> set[str]:
    """Lowercased words the LLM may introduce: every word of every known name and term."""
    names = [SHOW_WORDS, *hotwords]
    for c in glossary.champions:
        names += [c.name, *c.abilities]
    names += [i.name for i in glossary.items] + [s.term for s in glossary.slang]
    names += [n for c in glossary.concepts for n in [c.term, *c.aliases]]
    return {_token(w) for n in names for w in n.split() if _token(w)}


def _known(token: str, vocab: set[str]) -> bool:
    if "-" in token:
        return all(p in PREFIXES or _known(p, vocab) for p in token.split("-") if p)
    if token in vocab:
        return True
    for ending, dropped in SUFFIXES:
        if token.endswith(ending) and len(token) - len(ending) >= 2:
            stem = token[: -len(ending)]
            # "tankier" -> "tank", "jungling" -> "jungle", "ganked" -> "gank"; doubled
            # consonants too, as in "camping" -> "camp" or "gankker" -> "gank"
            if stem + dropped in vocab or (stem[-1:] == stem[-2:-1] and stem[:-1] in vocab):
                return True
    return False


def segment_ok(before: str, after: str, vocab: set[str]) -> bool:
    """True if every edit is at most MAX_SPAN words and is either punctuation or case only,
    or brings in only known words. Rules out moved, deleted, or reworded sentences."""
    a, b = before.split(), after.split()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if max(i2 - i1, j2 - j1) > MAX_SPAN:
            return False
        old = [t for t in map(_token, a[i1:i2]) if t]
        new = [t for t in map(_token, b[j1:j2]) if t]
        if old == new:
            continue
        if not new or not all(_known(t, vocab) for t in new):
            return False
    return True


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
