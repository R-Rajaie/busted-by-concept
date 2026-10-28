"""On-disk records written by the consolidate stage."""

from datetime import date

from pydantic import BaseModel

from coach.distill.models import Lesson


class Concept(BaseModel):
    """One concept card. Stored in `data/concepts/concepts.json`."""

    id: str  # "c0001"
    label: str  # canonical name, e.g. "wave management: when to freeze"
    lesson_ids: list[str]  # in date order
    stance: str  # the current view, from non-superseded lessons
    history: str  # one line on how the view changed, or "" if it never did
    host_stances: dict[str, str] = {}  # host id -> stance, only when hosts differ
    first_said: date | None = None
    last_reaffirmed: date | None = None
    centroid: list[float]  # mean lesson embedding, for assigning new lessons


class Consolidated(BaseModel):
    """Stored as `data/concepts/consolidated.json`.

    Lessons here are copies of `data/lessons/` with the consolidate fields filled in;
    the distill output itself is never modified.
    """

    threshold: float
    concepts: list[Concept]
    lessons: list[Lesson]
