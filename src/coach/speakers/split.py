"""Re-split diarized speakers that are two hosts merged into one -> `data/speaker_splits/`.

uv run python -m coach.speakers.split            # every suspect speaker
uv run python -m coach.speakers.split --episode <id>

Diarization sometimes clusters two hosts with similar voices (Curtis and Charlie) as one
speaker. Such a speaker matches one host but also resembles another host who is otherwise
missing from the episode. This cuts that speaker's speech into short stretches, embeds each
with the voiceprint model, and assigns each stretch to whichever of the two hosts it sounds
like. The result is a list of time ranges per label that override the speaker map; the speaker
maps and transcripts themselves are left as they are. Needs the `diarize` extra and the audio.
"""

import argparse
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from coach.config import AUDIO_DIR, DATA_DIR, SPEAKERS_DIR, TRANSCRIPTS_RAW_DIR
from coach.ingest.models import Episode
from coach.speakers.enroll import embed_chunks
from coach.speakers.identify import unit
from coach.speakers.models import SpeakerMap, load_config, load_voiceprints
from coach.transcribe.models import RawTranscript
from coach.transcribe.whisper import SAMPLE_RATE, load_audio

log = logging.getLogger("coach.speakers.split")

SPLITS_DIR = DATA_DIR / "speaker_splits"
SUSPECT = 0.45  # a matched speaker this similar to a missing host may contain them
MIN_SPEAKER_S = 600  # only speakers with this much talk time
STRETCH_MIN_S = 3.0  # shortest stretch worth embedding
STRETCH_MAX_S = 15.0
GAP_S = 0.8  # a pause this long between words starts a new stretch
MARGIN = 0.08  # a stretch is reassigned only if one host beats the other by this much


class Stretch(BaseModel):
    start: float
    end: float
    host: str  # "uncertain" when neither host wins by MARGIN
    similarity: dict[str, float]


class Split(BaseModel):
    label: str
    hosts: list[str]  # the matched host, then the one hidden inside
    stretches: list[Stretch]


class EpisodeSplits(BaseModel):
    """Stored as `data/speaker_splits/<episode_id>.json`."""

    episode_id: str
    splits: list[Split]


def suspects(m: SpeakerMap, active: list[str]) -> list[tuple[str, str, str]]:
    """(label, matched host, hidden host) for speakers that may be two hosts merged."""
    found = {s.host for s in m.speakers}
    out = []
    for s in m.speakers:
        if s.method != "voiceprint" or s.seconds < MIN_SPEAKER_S:
            continue
        for other, sim in sorted(s.similarity.items(), key=lambda kv: -kv[1]):
            if other != s.host and other in active and other not in found and sim >= SUSPECT:
                out.append((s.label, s.host, other))
                break
    return out


def stretches(raw: RawTranscript, label: str) -> list[tuple[float, float]]:
    """Runs of the label's words (no crosstalk), cut at pauses and at STRETCH_MAX_S."""
    out: list[tuple[float, float]] = []
    start = end = None
    for seg in raw.segments:
        for w in seg.words:
            mine = w.speaker == label and not w.overlap
            if (
                mine
                and start is not None
                and w.start - end <= GAP_S
                and w.end - start <= STRETCH_MAX_S
            ):
                end = w.end
                continue
            if start is not None and end - start >= STRETCH_MIN_S:
                out.append((start, end))
            start, end = (w.start, w.end) if mine else (None, None)
    if start is not None and end - start >= STRETCH_MIN_S:
        out.append((start, end))
    return out


def split_episode(eid: str, m: SpeakerMap, active: list[str], voiceprints: dict) -> EpisodeSplits:
    raw = RawTranscript.model_validate_json(
        (TRANSCRIPTS_RAW_DIR / f"{eid}.json").read_text("utf-8")
    )
    episode = Episode.model_validate_json((AUDIO_DIR / f"{eid}.json").read_text(encoding="utf-8"))
    audio, _ = load_audio(episode.audio_path(AUDIO_DIR), None)
    splits = []
    for label, host, hidden in suspects(m, active):
        spans = stretches(raw, label)
        if not spans:
            continue
        chunks = [audio[int(a * SAMPLE_RATE) : int(b * SAMPLE_RATE)] for a, b in spans]
        emb = embed_chunks(chunks)
        vp = {h: unit(voiceprints[h].embedding) for h in (host, hidden)}
        out = []
        for (a, b), e in zip(spans, emb, strict=True):
            sims = {h: round(float(e @ v), 4) for h, v in vp.items()}
            best, other = sorted(sims, key=sims.get, reverse=True)
            winner = best if sims[best] - sims[other] >= MARGIN else "uncertain"
            out.append(Stretch(start=a, end=b, host=winner, similarity=sims))
        secs = {
            h: sum(s.end - s.start for s in out if s.host == h) for h in (host, hidden, "uncertain")
        }
        log.info(
            "  %s (%s, may hide %s): %d stretches -> %s",
            label,
            host,
            hidden,
            len(out),
            ", ".join(f"{h} {v / 60:.1f} min" for h, v in secs.items()),
        )
        splits.append(Split(label=label, hosts=[host, hidden], stretches=out))
    return EpisodeSplits(episode_id=eid, splits=splits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.speakers.split", description=__doc__)
    parser.add_argument("--episode", action="append", metavar="ID", help="episode id (repeatable)")
    parser.add_argument("--force", action="store_true", help="redo episodes already split")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("speechbrain").setLevel(logging.WARNING)

    config = load_config()
    voiceprints = load_voiceprints(config)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(SPEAKERS_DIR.glob("*.json"))
    if args.episode:
        paths = [SPEAKERS_DIR / f"{e}.json" for e in args.episode]

    todo = []
    for p in paths:
        m = SpeakerMap.model_validate_json(p.read_text(encoding="utf-8"))
        meta = AUDIO_DIR / f"{m.episode_id}.json"
        aired = Episode.model_validate_json(meta.read_text(encoding="utf-8")).upload_date
        active = config.active(aired)
        if suspects(m, active) and (args.force or not (SPLITS_DIR / p.name).exists()):
            todo.append((m, active))
    log.info("%d episodes with suspect speakers", len(todo))
    for n, (m, active) in enumerate(todo, start=1):
        log.info("[%d/%d] %s", n, len(todo), m.episode_id)
        result = split_episode(m.episode_id, m, active, voiceprints)
        out = SPLITS_DIR / f"{m.episode_id}.json"
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return 0


def load_splits(episode_id: str, splits_dir: Path = SPLITS_DIR) -> dict[str, list[Stretch]]:
    """label -> stretches that override the speaker map, for one episode (empty if none)."""
    path = splits_dir / f"{episode_id}.json"
    if not path.exists():
        return {}
    done = EpisodeSplits.model_validate_json(path.read_text(encoding="utf-8"))
    return {s.label: s.stretches for s in done.splits}


if __name__ == "__main__":
    sys.exit(main())
