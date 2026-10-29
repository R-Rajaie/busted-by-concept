"""Enroll a host's voiceprint from clips of them speaking alone -> `hosts/<host>/voiceprint.json`.

uv run python -m coach.speakers.enroll --host curtis --url https://www.youtube.com/watch?v=...
uv run python -m coach.speakers.enroll --host nathan --url <url> --start 30 --end 900

The audio is kept under `hosts/<host>/reference/` (gitignored). Speech is cut into 10-second
chunks and embedded with the diarization pipeline's own embedding model; chunks that don't
sound like the rest (music, another voice) are dropped before averaging.
"""

import argparse
import logging
import re
import sys
from functools import cache
from pathlib import Path

import numpy as np

from coach import env
from coach.config import HOSTS_DIR
from coach.ingest.models import Episode
from coach.ingest.youtube import download_episode
from coach.speakers.identify import unit
from coach.speakers.models import Voiceprint
from coach.transcribe.diarize import EMBEDDING_MODEL
from coach.transcribe.whisper import SAMPLE_RATE, load_audio

log = logging.getLogger("coach.speakers.enroll")

CHUNK_S = 10
MIN_CHUNK_S = 5
KEEP = 0.6  # cosine to the mean; chunks below it are dropped as not the host
MIN_KEPT = 0.8  # warn if fewer than this share of chunks survive


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", url)
    if not m:
        raise ValueError(f"no YouTube video id in {url}")
    return m.group(1)


@cache
def load_embedder():
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(EMBEDDING_MODEL, use_auth_token=env.get("HF_TOKEN"))
    return Inference(model, window="whole")


def speech_chunks(audio: np.ndarray) -> list[np.ndarray]:
    """Cut VAD-detected speech into chunks of CHUNK_S seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    regions = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500))
    speech = np.concatenate([audio[r["start"] : r["end"]] for r in regions]) if regions else audio
    step = CHUNK_S * SAMPLE_RATE
    chunks = [speech[i : i + step] for i in range(0, len(speech), step)]
    return [c for c in chunks if len(c) >= MIN_CHUNK_S * SAMPLE_RATE]


def embed_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    import torch

    embedder = load_embedder()
    rows = [
        unit(embedder({"waveform": torch.from_numpy(c).unsqueeze(0), "sample_rate": SAMPLE_RATE}))
        for c in chunks
    ]
    return np.stack(rows)


def average(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(voiceprint, kept mask)`: the mean of the chunks that agree with the overall mean."""
    keep = embeddings @ unit(embeddings.mean(axis=0)) >= KEEP
    if not keep.any():
        raise ValueError("no chunks agree with each other; is this one speaker?")
    return unit(embeddings[keep].mean(axis=0)), keep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.speakers.enroll", description=__doc__)
    parser.add_argument("--host", required=True, help="host id, as in hosts/hosts.yaml")
    parser.add_argument("--url", action="append", required=True, help="solo clip (repeatable)")
    parser.add_argument("--start", type=float, default=0, help="skip the first N seconds")
    parser.add_argument("--end", type=float, help="stop at N seconds")
    parser.add_argument("--cookies", type=Path, help="cookies.txt for YouTube")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)

    ref_dir = HOSTS_DIR / args.host / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    chunks, ids = [], []
    for url in args.url:
        vid = video_id(url)
        meta = ref_dir / f"{vid}.json"
        if meta.exists():
            clip = Episode.model_validate_json(meta.read_text(encoding="utf-8"))
        else:
            log.info("downloading %s", url)
            clip = download_episode(vid, ref_dir, cookies=args.cookies)
            meta.write_text(clip.model_dump_json(indent=2), encoding="utf-8")
        audio, duration = load_audio(clip.audio_path(ref_dir), args.end)
        audio = audio[int(args.start * SAMPLE_RATE) :]
        found = speech_chunks(audio)
        log.info("%s (%.0f min): %d speech chunks", clip.title, duration / 60, len(found))
        chunks += found
        ids.append(vid)

    embeddings = embed_chunks(chunks)
    voiceprint, keep = average(embeddings)
    sims = embeddings @ voiceprint
    log.info(
        "kept %d/%d chunks; similarity to voiceprint min %.2f, median %.2f",
        keep.sum(),
        len(keep),
        sims[keep].min(),
        np.median(sims[keep]),
    )
    if keep.mean() < MIN_KEPT:
        log.warning("many chunks dropped; check the clip is mostly %s alone", args.host)

    out = HOSTS_DIR / args.host / "voiceprint.json"
    vp = Voiceprint(
        host=args.host,
        model=EMBEDDING_MODEL,
        source="enrolled",
        embedding=[float(x) for x in voiceprint],
        from_ids=ids,
        chunks=int(keep.sum()),
    )
    out.write_text(vp.model_dump_json(indent=2), encoding="utf-8")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
