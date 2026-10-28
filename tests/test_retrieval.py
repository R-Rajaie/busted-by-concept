from datetime import date

import numpy as np

from coach.consolidate import cluster
from coach.distill.models import Lesson
from coach.index.build import recency_ranks
from coach.index.search import staleness, weight


def lesson(n: int, day: date, **kw) -> Lesson:
    fields = dict(
        claim="c",
        reasoning="r",
        role="any",
        phase="lane",
        champions=[],
        durability="macro",
        start_ts=0,
        end_ts=1,
        id=f"e-{n:03d}",
        episode_id="e",
        episode_date=day,
        season="S16",
        speaker="uncertain",
    )
    return Lesson(**(fields | kw))


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_assign_joins_similar_and_splits_different():
    a, b, c = lesson(1, date(2025, 1, 20)), lesson(2, date(2025, 2, 1)), lesson(3, date(2025, 3, 1))
    vecs = {a.id: unit(1, 0), b.id: unit(0.95, 0.05), c.id: unit(0, 1)}
    concepts = []
    touched = cluster.assign([c, b, a], vecs, concepts, threshold=0.8)
    assert len(concepts) == 2 and touched == {"c0001", "c0002"}
    assert concepts[0].lesson_ids == [a.id, b.id]  # date order, regardless of input order


def test_recency_rank_is_within_concept_and_skips_superseded():
    old = lesson(1, date(2024, 3, 1), concept_id="c1")
    new = lesson(2, date(2026, 3, 1), concept_id="c1")
    gone = lesson(3, date(2023, 3, 1), concept_id="c1", status="superseded")
    lone_old = lesson(4, date(2022, 3, 1), concept_id="c2")
    ranks = recency_ranks([old, new, gone, lone_old])
    assert ranks == {old.id: 0.0, new.id: 1.0, lone_old.id: 1.0}


def test_staleness_depends_on_durability():
    today = date(2026, 6, 1)
    two_seasons_old = {"season": "S14", "durability": "patch"}
    assert staleness(two_seasons_old, today) == 0.25
    assert staleness({"season": "S14", "durability": "evergreen"}, today) > 0.9
    assert staleness({"season": "S16", "durability": "patch"}, today) == 1.0


def test_concept_cards_are_not_decayed():
    card = {"kind": "concept", "season": "", "durability": "", "recency_rank": 1.0}
    assert weight(card, 0.8) == 0.8
    older_in_concept = {
        "kind": "lesson",
        "season": "S16",
        "durability": "macro",
        "recency_rank": 0.0,
    }
    assert weight(older_in_concept, 0.8) == 0.4
