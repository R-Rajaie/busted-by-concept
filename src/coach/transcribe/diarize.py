"""Speaker diarization with pyannote, and assigning its labels to whisper's words.

Needs the `diarize` extra (`uv sync --extra diarize`) and `HF_TOKEN` in `.env`: the pipeline
is gated, so accept its terms on Hugging Face first. With `HF_HUB_OFFLINE=1` and the models
already in the Hugging Face cache (as on Kaggle), no token is needed.
"""

import logging
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from functools import cache

import numpy as np

from coach import env
from coach.transcribe.models import DiarizedSpeaker, Segment
from coach.transcribe.whisper import SAMPLE_RATE

log = logging.getLogger(__name__)

PIPELINE = "pyannote/speaker-diarization-3.1"
# The embedding model the 3.1 pipeline clusters with. Host voiceprints must use the same one,
# or their similarities to diarized speakers mean nothing.
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
BATCH_SIZE = 32


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@cache
def load_pipeline(device: str):
    import torch
    from pyannote.audio import Pipeline

    token = env.get("HF_TOKEN")
    if not token and not os.environ.get("HF_HUB_OFFLINE"):  # offline: models already cached
        raise RuntimeError(f"HF_TOKEN is not set; diarization needs it to download {PIPELINE}")
    log.info("loading %s (%s)", PIPELINE, device)
    pipeline = Pipeline.from_pretrained(PIPELINE, use_auth_token=token)
    pipeline.to(torch.device(device))
    if device == "cuda":  # the defaults of 1 leave most of the GPU idle
        pipeline.segmentation_batch_size = BATCH_SIZE
        pipeline.embedding_batch_size = BATCH_SIZE
    return pipeline


def diarize(audio: np.ndarray, device: str) -> tuple[list[Turn], list[DiarizedSpeaker]]:
    import torch

    waveform = torch.from_numpy(audio).unsqueeze(0)
    annotation, embeddings = load_pipeline(device)(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE}, return_embeddings=True
    )
    turns = [
        Turn(seg.start, seg.end, label) for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    speakers = []
    for label, emb in zip(annotation.labels(), embeddings, strict=False):
        usable = emb is not None and not np.isnan(emb).any()
        speakers.append(
            DiarizedSpeaker(
                label=label,
                seconds=annotation.label_duration(label),
                embedding=[float(x) for x in emb] if usable else None,
            )
        )
    return turns, speakers


def crosstalk(turns: list[Turn]) -> list[tuple[float, float]]:
    """Sorted, disjoint time ranges where two or more speakers' turns overlap."""
    # At equal times ends (-1) sort before starts (+1), so back-to-back turns don't count.
    events = sorted([(t.start, 1) for t in turns] + [(t.end, -1) for t in turns])
    out: list[tuple[float, float]] = []
    active, start = 0, 0.0
    for time, delta in events:
        before, active = active, active + delta
        if before < 2 <= active:
            start = time
        elif active < 2 <= before and time > start:
            out.append((start, time))
    return out


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> None:
    """Label each word with the speaker whose turns overlap it most, then each segment with
    its majority speaker by word time. Words no turn overlaps keep `speaker=None`.

    Words whose midpoint falls where pyannote heard two people at once get `overlap=True`:
    whisper writes down only one voice there, so who said them is a guess."""
    if not turns:
        return
    talk_over = crosstalk(turns)
    talk_over_starts = [a for a, _ in talk_over]
    turns = sorted(turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    longest = max(t.end - t.start for t in turns)
    for s in segments:
        lo, hi = bisect_left(starts, s.start - longest), bisect_left(starts, s.end)
        near = [t for t in turns[lo:hi] if t.end > s.start]
        by_speaker: dict[str, float] = defaultdict(float)
        for w in s.words:
            mid = (w.start + w.end) / 2
            i = bisect_right(talk_over_starts, mid) - 1
            w.overlap = i >= 0 and mid < talk_over[i][1]
            shared: dict[str, float] = defaultdict(float)
            for t in near:
                shared[t.speaker] += max(0.0, min(w.end, t.end) - max(w.start, t.start))
            best = max(shared, key=shared.__getitem__, default=None)
            if best is not None and shared[best] > 0:
                w.speaker = best
                by_speaker[best] += w.end - w.start
        s.speaker = max(by_speaker, key=by_speaker.__getitem__, default=None)
