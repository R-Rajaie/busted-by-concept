"""Draft `glossary/concepts.yaml`: the show's own named ideas, as the hosts define them.

uv run python -m coach.glossary.concepts                 # the episodes that define terms
uv run python -m coach.glossary.concepts --episode <id>  # others too (repeatable)
uv run python -m coach.glossary.concepts --term "weak side"  # one term, from the whole archive
uv run python -m coach.glossary.concepts --verify  # check each entry against its sources
uv run python -m coach.glossary.concepts --apply-reviews data/concept-reviews  # a person's answers

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


def cited_passages(concept: Concept, clean_dir: Path) -> str:
    cited = []
    for s in concept.sources[:5]:
        segs = _segments(s.episode_id, clean_dir)
        body = " ".join(
            t for start, t in segs if s.t - VERIFY_BEFORE_S <= start <= s.t + VERIFY_AFTER_S
        )
        cited.append(f"[{s.episode_id} @ {s.t}s] {body}")
    return "\n\n".join(cited)


def _entry(concept: Concept) -> str:
    fields = concept.model_dump(exclude={"sources", "reviewed"})
    return yaml.safe_dump(fields, sort_keys=False, allow_unicode=True)


def verify(concept: Concept, clean_dir: Path) -> Review:
    prompt = f"Entry:\n{_entry(concept)}\nCited passages:\n\n{cited_passages(concept, clean_dir)}"
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


REVISE = """You rewrite one entry in a glossary of the "Broken by Concept" League of Legends
podcast (hosts Curtis, Nathan, and Charlie). A person who knows the show reviewed it. Their
note is authoritative: it overrides the draft, the automated check, and your own reading of the
passages. Apply exactly what it says (a distinction to draw, a name to keep or drop, a fact to
fix). Otherwise stay strictly within what the cited passages support.

Return the corrected entry: term, aliases (other names the hosts actually use for this same
idea), and meaning (one to three sentences, standalone, in the hosts' terms). Set drop to true
only if the reviewer says to remove it or the passages show it isn't one of the show's ideas.
Background the reviewer gave: the hosts' coaching programs are Botlane Academy (BLA), Toplane
Academy (TLA), Midlane Academy (MLA) and Jungle Academy (formerly Saltu, said "saltoo"), and
their mentors mentor the academy's players, not the hosts."""


class _Revised(BaseModel):
    drop: bool = False
    term: str
    aliases: list[str] = []
    meaning: str


def revise(
    concept: Concept, check: dict | None, choice: str, note: str, clean_dir: Path
) -> Concept | None:
    """Rewrite one entry from its passages, the automated check, and the reviewer's note."""
    verdict = {
        "wrong": "The reviewer marked the current definition WRONG.",
        "suggested": "The reviewer accepted the suggested fix, with the note below.",
        "current": "The reviewer said the current definition is right, with the note below.",
    }[choice]
    prompt = (
        f"Entry:\n{_entry(concept)}\n"
        f"Automated check:\n{json.dumps(check or {}, ensure_ascii=False, indent=1)}\n\n"
        f"{verdict}\nReviewer's note: {note or '(none)'}\n\n"
        f"Cited passages:\n\n{cited_passages(concept, clean_dir)}"
    )
    got = llm.complete_model(prompt, REVISE, _Revised, model=llm.SMART_MODEL)
    if got.drop:
        return None
    return Concept(
        term=got.term,
        aliases=got.aliases,
        meaning=got.meaning,
        sources=concept.sources,
        reviewed="revised",
    )


def review_id(term: str, revised: bool = False) -> str:
    """The review page's id for an entry. A rewritten entry gets a fresh one, so the answer
    given to the old version doesn't count as a check of the new one."""
    slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")[:120] or "concept"
    return slug + "-rev" if revised else slug


def apply_reviews(reviews_dir: Path, path: Path, clean_dir: Path) -> int:
    """Apply a person's answers from the review page (exported as JSON files named by entry id).
    Entries already settled are left alone, so this can be rerun as more answers come in; a
    rewritten entry is settled again by the answer to its rewritten version."""
    concepts = [Concept(**c) for c in yaml.safe_load(path.read_text(encoding="utf-8")) or []]
    checks = json.loads(REVIEW_PATH.read_text(encoding="utf-8")) if REVIEW_PATH.exists() else {}
    answers = {}
    for f in reviews_dir.rglob("*.json"):
        doc = json.loads(f.read_text(encoding="utf-8"))
        data = doc.get("data", doc)
        if data.get("choice"):
            answers[doc.get("id", f.stem)] = data
    out, counts = [], Counter()
    for c in concepts:
        if c.reviewed in ("current", "suggested"):
            out.append(c)
            continue
        a = answers.get(review_id(c.term, revised=c.reviewed == "revised"))
        if a is None:
            out.append(c)
            continue
        confirmed = a["choice"] in ("current", "suggested") and not a.get("note")
        if c.reviewed == "revised" and confirmed:
            c.reviewed = "current"  # the rewrite was confirmed
            counts["confirmed"] += 1
            out.append(c)
            continue
        choice, note, check = a["choice"], (a.get("note") or "").strip(), checks.get(c.term)
        if choice == "wrong" or note:
            new = revise(c, check, choice, note, clean_dir)
            counts["dropped" if new is None else "revised"] += 1
            log.info("%s: %s", c.term, "dropped" if new is None else "revised")
            if new is not None:
                out.append(new)
            continue
        if choice == "suggested" and check and check.get("suggested_meaning"):
            wrong = {x.lower() for x in check.get("wrong_aliases", [])}
            c.meaning = check["suggested_meaning"]
            c.aliases = [x for x in c.aliases if x.lower() not in wrong]
        c.reviewed = choice
        counts[choice] += 1
        out.append(c)
    path.write_text(
        "# The show's own named ideas. Drafted by coach.glossary.concepts; review by hand.\n"
        + yaml.safe_dump(
            [c.model_dump(exclude_none=True) for c in out],
            sort_keys=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
    )
    log.info("applied %s; %d entries -> %s", dict(counts), len(out), path)
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
    parser.add_argument(
        "--apply-reviews",
        type=Path,
        metavar="DIR",
        help="apply review-page answers exported as JSON files in DIR",
    )
    parser.add_argument("--clean", type=Path, default=TRANSCRIPTS_CLEAN_DIR)
    parser.add_argument("--out", type=Path, default=GLOSSARY_DIR / "concepts.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.apply_reviews:
        return apply_reviews(args.apply_reviews, args.out, args.clean)
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
