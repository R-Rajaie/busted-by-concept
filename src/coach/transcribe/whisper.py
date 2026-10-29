"""faster-whisper wrapper. CPU (int8) by default; on a CUDA GPU it runs batched in float16."""

import logging
import os
from functools import cache
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import decode_audio

from coach.transcribe.models import Segment, Word

log = logging.getLogger(__name__)

DEFAULT_MODEL = "distil-large-v3"  # English-only, much faster than large-v3 on CPU
GPU_MODEL = "large-v3"  # more accurate on names; affordable once batched on a GPU
GPU_BATCH_SIZE = 16  # fits a 24 GB card with room for diarization
SAMPLE_RATE = 16_000
PROGRESS_EVERY_S = 300  # log every 5 minutes of audio


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    return device


@cache
def load_model(name: str, device: str = "cpu") -> WhisperModel:
    compute_type = "int8" if device == "cpu" else "float16"
    log.info("loading whisper model %s (%s, %s)", name, device, compute_type)
    return WhisperModel(
        name, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4
    )


def load_audio(path: Path, max_seconds: float | None = None) -> tuple[np.ndarray, float]:
    """Decode to 16 kHz mono once, so whisper and diarization share it.

    Returns `(audio, full_duration_s)`; the audio is cut to `max_seconds` if given.
    """
    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    if max_seconds is not None:
        audio = audio[: int(max_seconds * SAMPLE_RATE)]
    return audio, duration


def transcribe(
    audio: np.ndarray,
    model_name: str,
    hotwords: list[str],
    device: str = "cpu",
    batch_size: int = 1,
    name: str = "",
) -> tuple[list[Segment], str]:
    """Return `(segments, language)`. `batch_size > 1` uses the batched pipeline."""
    model = load_model(model_name, device)
    options = dict(
        language="en",
        vad_filter=True,
        word_timestamps=True,
        hotwords=", ".join(hotwords) or None,
        # Feeding previous text back in is the main cause of runaway repetition loops.
        condition_on_previous_text=False,
    )
    if batch_size > 1:
        raw, info = BatchedInferencePipeline(model).transcribe(
            audio, batch_size=batch_size, **options
        )
    else:
        raw, info = model.transcribe(audio, **options)
    segments, next_log = [], PROGRESS_EVERY_S
    total = len(audio) / SAMPLE_RATE
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
            log.info("  %s: %.0f / %.0f min", name, s.end / 60, total / 60)
            next_log += PROGRESS_EVERY_S
    return segments, info.language
