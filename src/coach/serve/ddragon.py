"""Live champion data from Data Dragon, cached per patch under `data/cache/ddragon/`."""

import json
import re
import urllib.request
from functools import cache
from typing import Any

from coach import glossary
from coach.config import DATA_DIR, DDRAGON_URL

CACHE_DIR = DATA_DIR / "cache" / "ddragon"


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


@cache
def latest_patch() -> str:
    return _get(f"{DDRAGON_URL}/api/versions.json")[0]


def _champion_json(champ_id: str, patch: str) -> dict:
    path = CACHE_DIR / patch / f"{champ_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = _get(f"{DDRAGON_URL}/cdn/{patch}/data/en_US/champion/{champ_id}.json")["data"][champ_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def find_champion_id(name: str) -> str | None:
    key = re.sub(r"[^a-z]", "", name.lower())
    for c in glossary.load().champions:
        if key in (re.sub(r"[^a-z]", "", c.name.lower()), c.id.lower()):
            return c.id
    return None


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def champion_summary(name: str) -> str:
    """Base stats, abilities with cooldowns and costs, and the current patch."""
    champ_id = find_champion_id(name)
    if champ_id is None:
        return f"No champion named {name!r}."
    patch = latest_patch()
    c = _champion_json(champ_id, patch)
    s = c["stats"]
    lines = [
        f"{c['name']}, {c['title']} (patch {patch}); classes: {', '.join(c['tags'])}",
        f"Resource: {c['partype']}",
        f"HP {s['hp']} (+{s['hpperlevel']}/lvl), armor {s['armor']} (+{s['armorperlevel']}), "
        f"MR {s['spellblock']} (+{s['spellblockperlevel']}), AD {s['attackdamage']} "
        f"(+{s['attackdamageperlevel']}), attack speed {s['attackspeed']}, "
        f"range {s['attackrange']}, move speed {s['movespeed']}",
        f"Passive: {c['passive']['name']}: {_strip_tags(c['passive']['description'])}",
    ]
    for key, spell in zip("QWER", c["spells"], strict=True):
        cds = "/".join(f"{x:g}" for x in spell["cooldown"])
        cost = "/".join(f"{x:g}" for x in spell["cost"])
        lines.append(
            f"{key}: {spell['name']} (cooldown {cds}s, cost {cost}): "
            f"{_strip_tags(spell['description'])}"
        )
    return "\n".join(lines)
