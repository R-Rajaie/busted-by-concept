"""Cluster lessons into concepts, label statuses, and write concept cards -> `data/concepts/`.

uv run python -m coach.consolidate --all              # rebuild every concept from scratch
uv run python -m coach.consolidate --episode <id>     # update only the concepts it touches
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from coach import embed
from coach.config import CONCEPTS_DIR, LESSONS_DIR
from coach.consolidate import cluster
from coach.consolidate.models import Concept, Consolidated
from coach.consolidate.review import review
from coach.distill.models import EpisodeLessons, Lesson

log = logging.getLogger("coach.consolidate")


def load_lessons(lessons_dir: Path, episode_ids: list[str] | None = None) -> list[Lesson]:
    paths = sorted(lessons_dir.glob("*.json"))
    if episode_ids is not None:
        paths = [lessons_dir / f"{e}.json" for e in episode_ids]
    out: list[Lesson] = []
    for p in paths:
        out += EpisodeLessons.model_validate_json(p.read_text(encoding="utf-8")).lessons
    return out


def lesson_text(x: Lesson) -> str:
    return f"{x.claim} {x.reasoning}"


class EmbeddingCache:
    """Lesson embeddings keyed by lesson id, stored as `embeddings.npz`."""

    def __init__(self, path: Path):
        self.path = path
        self.vecs: dict[str, np.ndarray] = {}
        if path.exists():
            data = np.load(path)
            self.vecs = dict(zip(data["ids"].tolist(), data["vecs"], strict=True))

    def ensure(self, lessons: list[Lesson], force: bool = False) -> None:
        todo = [x for x in lessons if force or x.id not in self.vecs]
        for x, v in zip(todo, embed.embed([lesson_text(x) for x in todo]), strict=True):
            self.vecs[x.id] = v

    def save(self) -> None:
        ids = list(self.vecs)
        np.savez(self.path, ids=np.array(ids), vecs=np.array([self.vecs[i] for i in ids]))


def apply_review(
    concept: Concept, by_id: dict[str, Lesson], concepts: list[Concept], vecs, depth: int = 0
) -> None:
    """Review one concept, split off misfits into their own concepts, and fill fields."""
    members = sorted((by_id[i] for i in concept.lesson_ids), key=_date_key)
    r = review(members)
    if r.misfit_ids and depth == 0:
        log.info("  %s: splitting off %d misfits", concept.id, len(r.misfit_ids))
        concept.lesson_ids = [i for i in concept.lesson_ids if i not in r.misfit_ids]
        for mid in r.misfit_ids:
            single = cluster.new_concept(concepts, by_id[mid], vecs[mid])
            apply_review(single, by_id, concepts, vecs, depth + 1)
        if concept.lesson_ids:
            cluster.recompute_centroid(concept, vecs)
            members = [m for m in members if m.id in concept.lesson_ids]
            r = review(members)  # statuses must be relative to the remaining lessons only

    statuses = {s.id: s for s in r.statuses}
    for m in members:
        s = statuses.get(m.id)
        m.concept_id = concept.id
        m.status = s.status if s else None
        m.superseded_by = s.superseded_by if s else None
        m.first_said = m.episode_date
    # A reinforcement reaffirms every earlier lesson that is still standing.
    for m in members:
        later = [
            x.episode_date
            for x in members
            if x.status == "reinforced"
            and x.episode_date
            and m.episode_date
            and x.episode_date >= m.episode_date
        ]
        m.last_reaffirmed = max([d for d in [m.episode_date, *later] if d], default=None)

    concept.lesson_ids = [m.id for m in members]
    concept.label = r.label
    concept.stance = r.stance
    concept.history = r.history
    concept.host_stances = {h.host: h.stance for h in r.host_stances}
    dates = [m.episode_date for m in members if m.episode_date]
    standing = [
        m.last_reaffirmed for m in members if m.status != "superseded" and m.last_reaffirmed
    ]
    concept.first_said = min(dates, default=None)
    concept.last_reaffirmed = max(standing, default=None)


def _date_key(x: Lesson) -> tuple:
    return (x.episode_date or x.id, x.start_ts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.consolidate", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="rebuild all concepts")
    target.add_argument("--episode", action="append", metavar="ID", help="episode id (repeatable)")
    parser.add_argument("--lessons", type=Path, default=LESSONS_DIR, help="input directory")
    parser.add_argument("--out", type=Path, default=CONCEPTS_DIR, help="output directory")
    parser.add_argument(
        "--threshold",
        type=float,
        default=cluster.DEFAULT_THRESHOLD,
        help="cosine similarity needed to join an existing concept",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "consolidated.json"
    cache = EmbeddingCache(args.out / "embeddings.npz")

    lost: set[str] = set()  # concepts that lost lessons and need a fresh review
    if args.all or not out_path.exists():
        lessons = load_lessons(args.lessons)
        concepts: list[Concept] = []
        new = lessons
        if not args.all:
            log.info("no existing concepts, so building from every lesson")
    else:
        prev = Consolidated.model_validate_json(out_path.read_text(encoding="utf-8"))
        concepts = prev.concepts
        new = load_lessons(args.lessons, args.episode)
        # Replace any earlier version of these episodes' lessons.
        stale = {x.id for x in prev.lessons if x.episode_id in set(args.episode)}
        lessons = [x for x in prev.lessons if x.id not in stale] + new
        for c in concepts:
            kept = [i for i in c.lesson_ids if i not in stale]
            if len(kept) < len(c.lesson_ids):
                lost.add(c.id)
            c.lesson_ids = kept
        concepts = [c for c in concepts if c.lesson_ids]

    log.info("embedding %d lessons", len(new))
    cache.ensure(new, force=True)
    by_id = {x.id: x for x in lessons}
    vecs = {i: cache.vecs[i] for i in by_id}

    touched = cluster.assign(new, vecs, concepts, args.threshold)
    touched |= {c.id for c in concepts if c.id in lost}
    log.info("%d lessons in %d concepts; reviewing %d", len(lessons), len(concepts), len(touched))

    for n, c in enumerate([c for c in concepts if c.id in touched], start=1):
        apply_review(c, by_id, concepts, vecs)
        log.info("[%d/%d] %s (%d lessons) %s", n, len(touched), c.id, len(c.lesson_ids), c.label)
    concepts = [c for c in concepts if c.lesson_ids]

    result = Consolidated(threshold=args.threshold, concepts=concepts, lessons=lessons)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(out_path)
    cache.save()
    log.info("wrote %d concepts", len(concepts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
