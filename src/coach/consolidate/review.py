"""LLM review of one concept: canonical label, per-lesson status, and the concept card."""

from pydantic import BaseModel, Field

from coach import llm
from coach.distill.models import Lesson, Status

SYSTEM = """You consolidate coaching lessons from a League of Legends podcast. You are given
lessons that an embedding model clustered into one concept, in date order.

1. Give the concept a short canonical label, "topic: specific question" style, e.g.
   "wave management: when to freeze".
2. List any lessons that clearly don't belong with the rest as misfits.
3. For each remaining lesson, set its status relative to the others:
   - reinforced: repeats an earlier lesson's point
   - refined: adds nuance to an earlier lesson (both stay valid)
   - superseded: a LATER lesson replaced this take (set superseded_by to that lesson id)
   - evolved: the same named host changed their view since an earlier lesson
   - disagreement: different named hosts hold conflicting views in the same era
   - null: the first statement of the idea, with nothing later replacing it
   If a lesson both reinforces an earlier one and is later replaced, use superseded.
   Speakers "uncertain" or "guest_unknown" are not identified, so never use evolved or
   disagreement for them; a contradicting later view from an unknown speaker is superseded.
4. Write the card: stance is the current view in 1-3 sentences, built from non-superseded
   lessons. history is one line on how the view changed, or "" if it never did.
   host_stances only when named hosts differ; otherwise empty."""


class LessonStatus(BaseModel):
    id: str
    status: Status | None
    superseded_by: str | None = None


class HostStance(BaseModel):
    host: str
    stance: str


class Review(BaseModel):
    label: str
    misfit_ids: list[str] = Field(description="lessons that don't belong in this concept")
    statuses: list[LessonStatus]
    stance: str
    history: str
    host_stances: list[HostStance] = []


def review(lessons: list[Lesson]) -> Review:
    lines = "\n".join(
        f"- id={x.id} date={x.episode_date} season={x.season} speaker={x.speaker} "
        f"durability={x.durability}\n  claim: {x.claim}\n  reasoning: {x.reasoning}"
        for x in lessons
    )
    model = llm.SMART_MODEL if len(lessons) > 1 else llm.FAST_MODEL
    result = llm.complete_model(f"Lessons:\n{lines}", SYSTEM, Review, model=model)
    ids = {x.id for x in lessons}
    # Drop anything the model invented; a single lesson can't be a misfit of itself.
    result.misfit_ids = [i for i in result.misfit_ids if i in ids] if len(lessons) > 1 else []
    result.statuses = [s for s in result.statuses if s.id in ids]
    for s in result.statuses:
        if s.superseded_by not in ids or s.superseded_by == s.id:
            s.superseded_by = None
        if s.status == "superseded" and s.superseded_by is None:
            s.status = None  # superseded with no replacement is meaningless
    # The model sometimes writes an empty history as a literal pair of quotes.
    result.history = result.history.strip().strip("\"'").strip()
    return result
