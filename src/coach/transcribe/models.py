"""On-disk records written by the transcribe stage."""

from pydantic import BaseModel


class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float
    speaker: str | None = None  # diarization label, e.g. "SPEAKER_01"
    overlap: bool = False  # said while another speaker was also talking (crosstalk)


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    speaker: str | None = None  # majority diarization label of its words; None if undiarized
    avg_logprob: float
    no_speech_prob: float
    suspect: bool = False  # flagged by the repeated-line detector; kept, not deleted
    words: list[Word] = []


class DiarizedSpeaker(BaseModel):
    """One anonymous speaker from diarization. Labels reset every episode."""

    label: str
    seconds: float  # total speaking time
    embedding: list[float] | None  # pyannote centroid; None when too little speech to embed


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
    speakers: list[DiarizedSpeaker] = []  # empty unless diarized
    segments: list[Segment]
