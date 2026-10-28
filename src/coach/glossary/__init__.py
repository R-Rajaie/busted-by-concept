"""Proper nouns and jargon used for hotwords and transcript correction.

`champions.json` and `items.json` are generated from Data Dragon by `coach.glossary.refresh`.
`slang.yaml` is hand-curated.
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


class Glossary(BaseModel):
    patch: str
    champions: list[Champion]
    items: list[Item]
    slang: list[SlangTerm]

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
    slang_path = glossary_dir / "slang.yaml"
    slang = yaml.safe_load(slang_path.read_text(encoding="utf-8")) if slang_path.exists() else []
    return Glossary(
        patch=champs["patch"],
        champions=champs["champions"],
        items=items["items"],
        slang=slang or [],
    )
