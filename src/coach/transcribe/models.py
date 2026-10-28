"""On-disk records written by the transcribe stage."""

from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None  # diarization label; None until diarization runs
    avg_logprob: float
    no_speech_prob: float
    suspect: bool = False  # flagged by the repeated-line detector; kept, not deleted
    words: list[Word] = []


class RawTranscript(BaseModel):
    """Stored as `data/transcripts/raw/<episode_id>.json`."""

    episode_id: str
    model: str
    language: str
    duration_s: float
    transcribed_s: float  # less than duration_s when partial
    partial: bool = False  # only part of the audio was transcribed (--max-seconds)
    diarized: bool = False
    hotwords: list[str]
    segments: list[Segment]
