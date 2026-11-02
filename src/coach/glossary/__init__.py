"""Proper nouns and jargon used for hotwords and transcript correction.

`champions.json` and `items.json` are generated from Data Dragon by `coach.glossary.refresh`.
`slang.yaml` is hand-curated. `concepts.yaml` holds the show's own named ideas, drafted from
the episodes that define them by `coach.glossary.concepts` and then reviewed by hand.
"""

import json
from functools import cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from coach.config import GLOSSARY_DIR


class Champion(BaseModel):
    id: str
    name: str
    title: str = ""
    tags: list[str] = []
    abilities: list[str] = []


class Item(BaseModel):
    id: str
    name: str
    plaintext: str = ""
    gold: int = 0


class SlangTerm(BaseModel):
    term: str
    misheard: list[str] = []
    meaning: str = ""
    core: bool = False  # always included in the hotword list


class Source(BaseModel):
    episode_id: str
    t: int  # seconds into the episode


class Concept(BaseModel):
    """One of the show's own named ideas ("mental stack", "3 blocking"), as the hosts define it."""

    term: str
    aliases: list[str] = []
    meaning: str
    sources: list[Source] = []


class Glossary(BaseModel):
    patch: str
    champions: list[Champion]
    items: list[Item]
    slang: list[SlangTerm]
    concepts: list[Concept] = []

    def terms(self) -> list[str]:
        """Every canonical spelling: champions, abilities, items, slang."""
        out = [c.name for c in self.champions]
        out += [a for c in self.champions for a in c.abilities]
        out += [i.name for i in self.items]
        out += [s.term for s in self.slang]
        return list(dict.fromkeys(out))


@cache
def load(glossary_dir: Path = GLOSSARY_DIR) -> Glossary:
    champs = json.loads((glossary_dir / "champions.json").read_text(encoding="utf-8"))
    items = json.loads((glossary_dir / "items.json").read_text(encoding="utf-8"))

    def hand_curated(name: str) -> list:
        path = glossary_dir / name
        return (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or []

    return Glossary(
        patch=champs["patch"],
        champions=champs["champions"],
        items=items["items"],
        slang=hand_curated("slang.yaml"),
        concepts=hand_curated("concepts.yaml"),
    )
