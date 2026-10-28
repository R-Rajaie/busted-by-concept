"""Regenerate `glossary/champions.json` and `glossary/items.json` from Data Dragon.

uv run python -m coach.glossary.refresh
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path
from typing import Any

from coach.config import DDRAGON_URL, GLOSSARY_DIR
from coach.glossary import Champion, Item

log = logging.getLogger("coach.glossary")

SUMMONERS_RIFT = "11"


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def fetch_champions(patch: str) -> list[Champion]:
    data = _get(f"{DDRAGON_URL}/cdn/{patch}/data/en_US/championFull.json")["data"]
    champs = []
    for c in data.values():
        abilities = [c["passive"]["name"]] + [s["name"] for s in c["spells"]]
        champs.append(
            Champion(
                id=c["id"], name=c["name"], title=c["title"], tags=c["tags"], abilities=abilities
            )
        )
    return sorted(champs, key=lambda c: c.name)


def fetch_items(patch: str) -> list[Item]:
    data = _get(f"{DDRAGON_URL}/cdn/{patch}/data/en_US/item.json")["data"]
    by_name: dict[str, Item] = {}
    for item_id, i in data.items():
        if not i["maps"].get(SUMMONERS_RIFT) or not i["gold"]["purchasable"]:
            continue
        # Data Dragon has duplicate entries (e.g. Ornn upgrades, arena copies); keep the first.
        by_name.setdefault(
            i["name"],
            Item(
                id=item_id,
                name=i["name"],
                plaintext=i.get("plaintext", ""),
                gold=i["gold"]["total"],
            ),
        )
    return sorted(by_name.values(), key=lambda i: i.name)


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.glossary.refresh", description=__doc__)
    parser.add_argument("--patch", help="Data Dragon version (default: latest)")
    parser.add_argument("--out", type=Path, default=GLOSSARY_DIR, help="output directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    patch = args.patch or _get(f"{DDRAGON_URL}/api/versions.json")[0]
    log.info("Data Dragon patch %s", patch)

    champions = fetch_champions(patch)
    items = fetch_items(patch)
    args.out.mkdir(parents=True, exist_ok=True)
    write(
        args.out / "champions.json",
        {"patch": patch, "champions": [c.model_dump() for c in champions]},
    )
    write(args.out / "items.json", {"patch": patch, "items": [i.model_dump() for i in items]})
    log.info("wrote %d champions, %d items", len(champions), len(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
