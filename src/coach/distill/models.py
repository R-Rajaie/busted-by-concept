"""On-disk records written by the distill stage (and extended by consolidate)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["top", "jungle", "mid", "adc", "support", "any"]
Phase = Literal["draft", "lane", "mid", "late", "teamfight", "mental"]
Durability = Literal["patch", "macro", "evergreen"]
Status = Literal["reinforced", "refined", "superseded", "evolved", "disagreement"]


class ExtractedLesson(BaseModel):
    """What the LLM fills in. Episode-level fields are added in code, not by the model."""

    claim: str = Field(description="the advice itself, one or two sentences, self-contained")
    reasoning: str = Field(description="why the hosts say it works, in their terms")
    role: Role
    phase: Phase
    champions: list[str] = Field(description="champions the lesson is specifically about")
    durability: Durability = Field(
        description="patch: depends on current stats/items/meta; macro: strategy that shifts "
        "slowly; evergreen: fundamentals and mindset"
    )
    start_ts: float = Field(description="seconds, from the transcript timestamps")
    end_ts: float = Field(description="seconds, from the transcript timestamps")


class Lesson(ExtractedLesson):
    id: str  # "<episode_id>-<n>"
    episode_id: str
    episode_date: date | None
    season: str | None  # e.g. "S14"; None means unmapped, see season_flag
    season_flag: str | None = None  # why the season couldn't be mapped
    speaker: str  # host id, "guest_unknown", or "uncertain"

    # Set by the consolidate stage.
    concept_id: str | None = None
    status: Status | None = None
    superseded_by: str | None = None
    first_said: date | None = None
    last_reaffirmed: date | None = None


class EpisodeLessons(BaseModel):
    """Stored as `data/lessons/<episode_id>.json`."""

    episode_id: str
    partial: bool  # copied from the clean transcript
    lessons: list[Lesson]
