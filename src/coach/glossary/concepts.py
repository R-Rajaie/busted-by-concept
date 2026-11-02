"""Draft `glossary/concepts.yaml`: the show's own named ideas, as the hosts define them.

uv run python -m coach.glossary.concepts                 # the episodes that define terms
uv run python -m coach.glossary.concepts --episode <id>  # others too (repeatable)
uv run python -m coach.glossary.concepts --term "weak side"  # one term, from the whole archive
uv run python -m coach.glossary.concepts --verify  # check each entry against its sources

Reads clean transcripts in 10-minute windows and asks the smart model for every concept the
hosts name and explain, then one more call merges duplicates across windows and episodes into
one entry per concept. The output is a draft: review it, since the rest of the pipeline
(distill, the coach) treats it as the show's vocabulary. Existing entries are kept as they are
and new ones are added after them, so hand edits survive a rerun.
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from coach import llm
from coach.config import DATA_DIR, GLOSSARY_DIR, TRANSCRIPTS_CLEAN_DIR, TRANSCRIPTS_RAW_DIR
from coach.correct.models import CleanTranscript
from coach.distill.__main__ import windows
from coach.glossary import Concept, Source

log = logging.getLogger("coach.glossary.concepts")

REVIEW_PATH = DATA_DIR / "concepts-review.json"

# Episodes where the hosts set out to define their vocabulary.
DEFINING = [
    "Jbw9Dq4GWUc",  # Official BBC Dictionary Episode (#239)
    "ANlsvzO-fdc",  # Inconsistency Doesn't Exist & Dictionary Part 2 (#240)
    "2WQ8zHtK6fc",  # The Complete A-Z of League of Legends Improvement (#268)
    "EgAO9qiKYm4",  # Every Term in League of Legends Explained
    "G1A0U5bJRZM",  # Complete MENTAL STACK Guide
    "1AO2iZwUt9s",  # The BBC Pyramid (Stage 1-4) (#144)
    "uZQvmLrUVKM",  # The Four Stages of Improving at League of Legends
    "rWp2ZO2zHy4",  # Ep 089: Alternatives To 3 Blocking
]

FIND = """You read a window of a League of Legends coaching podcast ("Broken by Concept",
hosts Curtis, Nathan, and Charlie) and list the named concepts the hosts define or explain.

A concept is a term, framework, or named idea the hosts use with a specific meaning: their own
coinages ("mental stack", "3 blocking", "champion reference points") and general words they
give a particular sense ("intensity", "deploys"). Skip plain game terms that mean the same
thing everywhere (CS, gank, Baron) unless the hosts add their own twist to them.

For each, give the term as the hosts say it, other names they use for it, what it means in
their words (one to three sentences, standalone), and the [seconds] timestamp where they
explain it. Return an empty list if the window defines nothing."""

MERGE = """You merge a list of concept definitions pulled from several windows and episodes of
the "Broken by Concept" League of Legends podcast. Combine entries that name the same idea
(including different spellings or aliases) into one, choosing the name the hosts use most.
Write one definition per concept in the hosts' terms, one to three sentences, drawing on all
the merged entries. Keep every source of every merged entry. Drop entries that are only plain
game terms with no special meaning on the show."""


DEFINE = """You read passages from a League of Legends coaching podcast ("Broken by Concept",
hosts Curtis, Nathan, and Charlie) where a given term comes up, each tagged [episode @ seconds].
Decide whether the hosts use the term as a concept with a specific meaning (their own coinage,
a framework, or a game idea they explain), and if so define it as they use it: one to three
sentences, standalone, in their terms, with other names they use for it. Set sources to the
passages that best explain it (episode id and seconds). If they only say the word in passing,
or it means nothing beyond its plain sense, set is_concept to false and say why."""

VERIFY = """You check one entry in a glossary of the "Broken by Concept" League of Legends
podcast (hosts Curtis, Nathan, and Charlie) against the transcript passages it cites, each
tagged [episode @ seconds]. Be strict: the entry will teach a coaching assistant what the hosts
mean, so a confident wrong definition is worse than a vague one.

Report:
- problems: each claim in the meaning that the passages don't support or that contradicts them,
  and any place it merges this idea with a different one the hosts treat separately.
- wrong_aliases: listed other names that are really a different idea, or that the hosts never
  use for this one.
- verdict: "ok" if the entry is right as written, "fix" if it has problems you can correct from
  the passages, "unsupported" if the passages don't show what the hosts mean by the term.
- suggested_meaning: for "fix", a corrected meaning (one to three sentences, standalone, in the
  hosts' terms, using only what the passages support); otherwise empty."""

VERIFY_BEFORE_S = 60  # passage window around each cited timestamp
VERIFY_AFTER_S = 150
VERIFY_WORKERS = 3  # parallel claude calls
TERM_EPISODES = 6  # episodes that say the term most
TERM_WINDOWS = 3  # passages per episode


class _Defined(BaseModel):
    is_concept: bool
    why: str = ""
    concept: Concept | None = None


class Review(BaseModel):
    """The model's check of one concept against its sources, for a person to confirm."""

    term: str
    verdict: Literal["ok", "fix", "unsupported"]
    problems: list[str] = []
    wrong_aliases: list[str] = []
    suggested_meaning: str = ""


class _Found(BaseModel):
    term: str
    aliases: list[str] = []
    meaning: str
    start_ts: float = Field(description="seconds, from the transcript timestamps")


class _FoundList(BaseModel):
    concepts: list[_Found]


class _Merged(BaseModel):
    concepts: list[Concept]


def find(clean: CleanTranscript) -> list[Concept]:
    found = []
    chunks = windows(clean.segments)
    for w, chunk in enumerate(chunks, start=1):
        lines = "\n".join(f"[{s.start:.0f}] {s.text}" for s in chunk)
        got = llm.complete_model(lines, FIND, _FoundList, model=llm.SMART_MODEL)
        for c in got.concepts:
            found.append(
                Concept(
                    term=c.term,
                    aliases=c.aliases,
                    meaning=c.meaning,
                    sources=[Source(episode_id=clean.episode_id, t=int(c.start_ts))],
                )
            )
        log.info("  window %d/%d: %d concepts", w, len(chunks), len(got.concepts))
    return found


def merge(found: list[Concept]) -> list[Concept]:
    payload = yaml.safe_dump([c.model_dump() for c in found], sort_keys=False, allow_unicode=True)
    return llm.complete_model(payload, MERGE, _Merged, model=llm.SMART_MODEL).concepts


def _segments(eid: str, clean_dir: Path) -> list[tuple[float, str]]:
    """(start, text) for an episode, from its clean transcript if there is one yet."""
    clean = clean_dir / f"{eid}.json"
    path = clean if clean.exists() else TRANSCRIPTS_RAW_DIR / f"{eid}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(s["start"], s["text"]) for s in data["segments"]]


def passages(term: str, clean_dir: Path) -> list[tuple[str, float, str]]:
    """(episode, start, text) windows around the term, from the episodes that say it most."""
    words = [re.escape(w) for w in re.split(r"[\s-]+", term.strip())]
    pattern = re.compile(r"\b" + r"[\s-]*".join(words) + r"\w*", re.IGNORECASE)
    hits: dict[str, list[int]] = {}
    episodes: dict[str, list[tuple[float, str]]] = {}
    for path in TRANSCRIPTS_RAW_DIR.glob("*.json"):
        segs = _segments(path.stem, clean_dir)
        found = [i for i, (_, t) in enumerate(segs) if pattern.search(t)]
        if found:
            hits[path.stem], episodes[path.stem] = found, segs
    out = []
    for eid in sorted(hits, key=lambda e: -len(hits[e]))[:TERM_EPISODES]:
        segs, last = episodes[eid], -1
        for i in hits[eid][:TERM_WINDOWS]:
            if i <= last:
                continue  # already inside the previous window
            lo, last = max(0, i - 4), min(len(segs), i + 8)
            out.append((eid, segs[lo][0], " ".join(t for _, t in segs[lo:last])))
    return out


def define(term: str, clean_dir: Path) -> Concept | None:
    """Define one named term from wherever the hosts talk about it, or None if they don't."""
    found = passages(term, clean_dir)
    if not found:
        log.info("%r: never said", term)
        return None
    text = "\n\n".join(f"[{eid} @ {t:.0f}s] {body}" for eid, t, body in found)
    got = llm.complete_model(f"Term: {term}\n\n{text}", DEFINE, _Defined, model=llm.SMART_MODEL)
    if not got.is_concept or got.concept is None:
        log.info("%r: not a concept on the show (%s)", term, got.why)
        return None
    log.info("%r: defined from %d passages", term, len(found))
    return got.concept


def verify(concept: Concept, clean_dir: Path) -> Review:
    cited = []
    for s in concept.sources[:5]:
        segs = _segments(s.episode_id, clean_dir)
        body = " ".join(
            t for start, t in segs if s.t - VERIFY_BEFORE_S <= start <= s.t + VERIFY_AFTER_S
        )
        cited.append(f"[{s.episode_id} @ {s.t}s] {body}")
    entry = yaml.safe_dump(
        concept.model_dump(exclude={"sources"}), sort_keys=False, allow_unicode=True
    )
    prompt = f"Entry:\n{entry}\nCited passages:\n\n" + "\n\n".join(cited)
    review = llm.complete_model(prompt, VERIFY, Review, model=llm.SMART_MODEL)
    review.term = concept.term
    return review


def verify_all(concepts: list[Concept], clean_dir: Path, out: Path) -> int:
    """Check every concept against its sources, in parallel, resuming from `out`."""
    done: dict[str, dict] = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo = [c for c in concepts if c.term not in done]
    log.info("verifying %d concepts (%d already done)", len(todo), len(done))
    with ThreadPoolExecutor(VERIFY_WORKERS) as pool:
        futures = {pool.submit(verify, c, clean_dir): c for c in todo}
        for n, f in enumerate(as_completed(futures), start=1):
            c = futures[f]
            try:
                done[c.term] = f.result().model_dump()
            except llm.LLMStop:
                pool.shutdown(cancel_futures=True)
                raise
            except Exception as e:  # keep going; a rerun picks it up
                log.error("%s failed: %s", c.term, e)
                continue
            out.write_text(json.dumps(done, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("[%d/%d] %s: %s", n, len(todo), c.term, done[c.term]["verdict"])
    verdicts = Counter(r["verdict"] for r in done.values())
    log.info("%s -> %s", dict(verdicts), out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.glossary.concepts", description=__doc__)
    parser.add_argument("--episode", action="append", metavar="ID", help="also read this one")
    parser.add_argument(
        "--term", action="append", help="only define this term, from the whole archive (repeatable)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every entry against its sources -> data/concepts-review.json",
    )
    parser.add_argument("--clean", type=Path, default=TRANSCRIPTS_CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=GLOSSARY_DIR / "concepts.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.verify:
        saved = yaml.safe_load(args.out.read_text(encoding="utf-8")) or []
        return verify_all([Concept(**c) for c in saved], args.clean, REVIEW_PATH)

    if args.term:
        merged = [c for c in (define(t, args.clean) for t in args.term) if c is not None]
    else:
        found: list[Concept] = []
        for eid in DEFINING + (args.episode or []):
            path = args.clean / f"{eid}.json"
            if not path.exists():
                log.error("%s has no clean transcript; run coach.correct --episode %s", eid, eid)
                return 1
            log.info("%s", eid)
            found += find(CleanTranscript.model_validate_json(path.read_text(encoding="utf-8")))
        log.info("merging %d raw entries", len(found))
        merged = merge(found)

    existing = []
    if args.out.exists():
        saved = yaml.safe_load(args.out.read_text(encoding="utf-8")) or []
        existing = [Concept(**c) for c in saved]
    known = {n.lower() for c in existing for n in [c.term, *c.aliases]}
    added = [c for c in merged if c.term.lower() not in known]
    out = [c.model_dump() for c in existing + added]
    args.out.write_text(
        "# The show's own named ideas. Drafted by coach.glossary.concepts; review by hand.\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    log.info("%d concepts (%d new) -> %s", len(out), len(added), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
