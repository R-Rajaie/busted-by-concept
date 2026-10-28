"""Transcribe downloaded episodes to `data/transcripts/raw/`.

uv run python -m coach.transcribe --episode <id>
uv run python -m coach.transcribe --all --limit 3
uv run python -m coach.transcribe --episode <id> --max-seconds 600   # quick partial run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from coach import glossary
from coach.config import AUDIO_DIR, TRANSCRIPTS_RAW_DIR
from coach.ingest.models import Episode
from coach.transcribe import hallucination, hotwords
from coach.transcribe.models import RawTranscript
from coach.transcribe.whisper import DEFAULT_MODEL, transcribe

log = logging.getLogger("coach.transcribe")


def is_transcribed(episode_id: str, out_dir: Path) -> bool:
    path = out_dir / f"{episode_id}.json"
    if not path.exists():
        return False
    return not RawTranscript.model_validate_json(path.read_text(encoding="utf-8")).partial


def ingested_episodes(audio_dir: Path) -> list[Episode]:
    episodes = []
    for meta in audio_dir.glob("*.json"):
        episode = Episode.model_validate_json(meta.read_text(encoding="utf-8"))
        if episode.audio_path(audio_dir).exists():
            episodes.append(episode)
    return sorted(episodes, key=lambda e: e.playlist_index or 0)


def write_transcript(transcript: RawTranscript, out_dir: Path) -> None:
    path = out_dir / f"{transcript.episode_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.transcribe", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every ingested episode")
    target.add_argument("--episode", action="append", metavar="ID", help="episode id (repeatable)")
    parser.add_argument("--audio", type=Path, default=AUDIO_DIR, help="ingest output directory")
    parser.add_argument("--out", type=Path, default=TRANSCRIPTS_RAW_DIR, help="output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model name")
    parser.add_argument("--force", action="store_true", help="re-transcribe existing episodes")
    parser.add_argument("--limit", type=int, help="stop after this many new transcripts")
    parser.add_argument("--max-seconds", type=float, help="only transcribe the first N seconds")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # model download chatter
    args.out.mkdir(parents=True, exist_ok=True)
    gloss = glossary.load()

    episodes = ingested_episodes(args.audio)
    if args.episode:
        wanted = set(args.episode)
        episodes = [e for e in episodes if e.episode_id in wanted]
        missing = wanted - {e.episode_id for e in episodes}
        if missing:
            log.error("not ingested: %s", " ".join(sorted(missing)))
            return 1

    done, skipped, failed = 0, 0, []
    for n, episode in enumerate(episodes, start=1):
        tag = f"[{n}/{len(episodes)}] {episode.episode_id}"
        if not args.force and is_transcribed(episode.episode_id, args.out):
            skipped += 1
            continue
        if args.limit is not None and done >= args.limit:
            log.info("hit --limit %d, stopping", args.limit)
            break
        words = hotwords.build(episode, gloss)
        log.info("%s transcribing %s (%d hotwords)", tag, episode.title, len(words))
        started = time.monotonic()
        try:
            segments, language, duration = transcribe(
                episode.audio_path(args.audio), args.model, words, args.max_seconds
            )
        except Exception as e:  # keep going; a rerun picks failures back up
            log.error("%s failed: %s", tag, e)
            failed.append(episode.episode_id)
            continue
        suspect = hallucination.flag(segments)
        transcribed = min(duration, args.max_seconds) if args.max_seconds else duration
        write_transcript(
            RawTranscript(
                episode_id=episode.episode_id,
                model=args.model,
                language=language,
                duration_s=duration,
                transcribed_s=transcribed,
                partial=transcribed < duration,
                hotwords=words,
                segments=segments,
            ),
            args.out,
        )
        done += 1
        elapsed = time.monotonic() - started
        log.info(
            "%s done: %d segments, %d suspect, %.1fx realtime",
            tag,
            len(segments),
            suspect,
            transcribed / elapsed,
        )

    log.info("transcribed %d, skipped %d existing, failed %d", done, skipped, len(failed))
    if failed:
        log.error("failed ids: %s", " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
