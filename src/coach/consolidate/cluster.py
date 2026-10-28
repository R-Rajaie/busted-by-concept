"""Assign lessons to concepts by embedding similarity.

This is incremental leader clustering: each lesson, in date order, joins the concept whose
centroid is most similar if that clears the threshold, else starts a new concept. It's
order-dependent and imperfect, which is why the LLM review can split off misfits.
"""

import numpy as np

from coach.consolidate.models import Concept
from coach.distill.models import Lesson

DEFAULT_THRESHOLD = 0.80


def next_id(concepts: list[Concept]) -> str:
    n = max((int(c.id[1:]) for c in concepts), default=0)
    return f"c{n + 1:04d}"


def new_concept(concepts: list[Concept], lesson: Lesson, vec: np.ndarray) -> Concept:
    concept = Concept(
        id=next_id(concepts),
        label="",
        lesson_ids=[lesson.id],
        stance="",
        history="",
        centroid=vec.tolist(),
    )
    concepts.append(concept)
    return concept


def recompute_centroid(concept: Concept, vecs: dict[str, np.ndarray]) -> None:
    mean = np.mean([vecs[i] for i in concept.lesson_ids], axis=0)
    concept.centroid = (mean / np.linalg.norm(mean)).tolist()


def assign(
    lessons: list[Lesson],
    vecs: dict[str, np.ndarray],
    concepts: list[Concept],
    threshold: float = DEFAULT_THRESHOLD,
) -> set[str]:
    """Add each lesson to a concept, mutating `concepts`. Returns the touched concept ids."""
    touched: set[str] = set()
    for lesson in sorted(lessons, key=lambda x: (x.episode_date or x.id, x.start_ts)):
        vec = vecs[lesson.id]
        best, best_sim = None, threshold
        for c in concepts:
            sim = float(np.dot(vec, c.centroid))
            if sim >= best_sim:
                best, best_sim = c, sim
        if best is None:
            best = new_concept(concepts, lesson, vec)
        else:
            best.lesson_ids.append(lesson.id)
            recompute_centroid(best, vecs)
        touched.add(best.id)
    return touched
