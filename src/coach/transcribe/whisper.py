"""faster-whisper wrapper. Runs on CPU (int8) since this machine has no CUDA GPU."""

import logging
import os
from functools import cache
from pathlib import Path

from faster_whisper import WhisperModel

from coach.transcribe.models import Segment, Word

log = logging.getLogger(__name__)

DEFAULT_MODEL = "distil-large-v3"  # English-only, much faster than large-v3 on CPU
PROGRESS_EVERY_S = 300  # log every 5 minutes of audio


@cache
def load_model(name: str, device: str = "cpu", compute_type: str = "int8") -> WhisperModel:
    log.info("loading whisper model %s (%s, %s)", name, device, compute_type)
    return WhisperModel(
        name, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4
    )


def transcribe(
    audio: Path,
    model_name: str,
    hotwords: list[str],
    max_seconds: float | None = None,
) -> tuple[list[Segment], str, float]:
    """Return `(segments, language, audio_duration_s)`."""
    model = load_model(model_name)
    kwargs = {}
    if max_seconds is not None:
        kwargs["clip_timestamps"] = f"0,{max_seconds}"
    raw, info = model.transcribe(
        str(audio),
        language="en",
        vad_filter=True,
        word_timestamps=True,
        hotwords=", ".join(hotwords) or None,
        # Feeding previous text back in is the main cause of runaway repetition loops.
        condition_on_previous_text=False,
        **kwargs,
    )
    segments, next_log = [], PROGRESS_EVERY_S
    for i, s in enumerate(raw):
        segments.append(
            Segment(
                id=i,
                start=s.start,
                end=s.end,
                text=s.text.strip(),
                avg_logprob=s.avg_logprob,
                no_speech_prob=s.no_speech_prob,
                words=[
                    Word(start=w.start, end=w.end, text=w.word, prob=w.probability)
                    for w in s.words or []
                ],
            )
        )
        if s.end >= next_log:
            log.info("  %s: %.0f / %.0f min", audio.stem, s.end / 60, info.duration / 60)
            next_log += PROGRESS_EVERY_S
    return segments, info.language, info.duration
