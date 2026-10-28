"""On-disk records written by the correct stage."""

from typing import Literal

from pydantic import BaseModel

from coach.transcribe.models import Segment


class CleanSegment(Segment):
    raw_text: str  # transcribe-stage text, before any correction


class Correction(BaseModel):
    segment_id: int
    before: str
    after: str
    source: Literal["slang", "fuzzy", "llm"]


class RejectedChunk(BaseModel):
    first_segment: int
    last_segment: int
    change_ratio: float


class CleanTranscript(BaseModel):
    """Stored as `data/transcripts/clean/<episode_id>.json`."""

    episode_id: str
    partial: bool  # copied from the raw transcript
    segments: list[CleanSegment]
    corrections: list[Correction]
    rejected_chunks: list[RejectedChunk] = []
