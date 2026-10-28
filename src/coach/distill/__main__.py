"""Extract structured lessons from clean transcripts -> `data/lessons/`.

uv run python -m coach.distill --all
uv run python -m coach.distill --episode <id>
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from coach import llm
from coach.config import AUDIO_DIR, LESSONS_DIR, TRANSCRIPTS_CLEAN_DIR
from coach.correct.models import CleanSegment, CleanTranscript
from coach.distill.models import EpisodeLessons, ExtractedLesson, Lesson
from coach.ingest.models import Episode

log = logging.getLogger("coach.distill")

WINDOW_S = 600  # transcript window per LLM call

SYSTEM = """You extract coaching lessons from a League of Legends improvement podcast.

A lesson is concrete, actionable advice or a principle the hosts endorse, which a player
could apply in their own games. Skip banter, sponsor reads, intros, recaps of earlier
points in the same window, and claims the hosts are only describing to disagree with.

Write each claim so it stands alone without the transcript. Keep the hosts' meaning; don't
add advice they didn't give. Use role "any" and an empty champions list unless the lesson is
really specific to a role or champions. Use the transcript's [seconds] timestamps for
start_ts and end_ts. Return an empty list if the window has no lessons."""


class _Extracted(BaseModel):
    lessons: list[ExtractedLesson]


def season_for(upload: date | None) -> tuple[str | None, str | None]:
    """Map an upload date to a season label, or return a flag explaining why not."""
    if upload is None:
        return None, "no upload date"
    # Seasons have started in early-to-mid January since S3 (2013); S14 = 2024.
    if upload.year < 2013:
        return None, f"upload date {upload} predates season mapping"
    if upload.month == 1 and upload.day < 15:
        return None, f"upload date {upload} is near a season boundary"
    return f"S{upload.year - 2010}", None


def windows(segments: list[CleanSegment], size: float = WINDOW_S) -> list[list[CleanSegment]]:
    out: list[list[CleanSegment]] = []
    for s in segments:
        if not out or s.start - out[-1][0].start >= size:
            out.append([])
        out[-1].append(s)
    return out


def distill(clean: CleanTranscript, episode: Episode) -> EpisodeLessons:
    season, flag = season_for(episode.upload_date)
    lessons: list[Lesson] = []
    chunks = windows(clean.segments)
    for w, chunk in enumerate(chunks, start=1):
        lines = "\n".join(f"[{s.start:.0f}] {s.text}" for s in chunk)
        prompt = (
            f"Episode: {episode.title}\nUploaded: {episode.upload_date}\n\n"
            f"Transcript window {w}/{len(chunks)}:\n{lines}"
        )
        extracted = llm.complete_model(prompt, SYSTEM, _Extracted, model=llm.SMART_MODEL)
        for x in extracted.lessons:
            lessons.append(
                Lesson(
                    **x.model_dump(),
                    id=f"{episode.episode_id}-{len(lessons) + 1:03d}",
                    episode_id=episode.episode_id,
                    episode_date=episode.upload_date,
                    season=season,
                    season_flag=flag,
                    # No diarization yet, so nothing can be attributed to a host.
                    speaker="uncertain",
                )
            )
        log.info("  window %d/%d: %d lessons", w, len(chunks), len(extracted.lessons))
    return EpisodeLessons(episode_id=clean.episode_id, partial=clean.partial, lessons=lessons)


def is_distilled(clean: CleanTranscript, out_dir: Path) -> bool:
    path = out_dir / f"{clean.episode_id}.json"
    if not path.exists():
        return False
    done = EpisodeLessons.model_validate_json(path.read_text(encoding="utf-8"))
    return done.partial == clean.partial


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.distill", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every clean transcript")
    target.add_argument("--episode", action="append", metavar="ID", help="episode id (repeatable)")
    parser.add_argument("--clean", type=Path, default=TRANSCRIPTS_CLEAN_DIR, help="input dir")
    parser.add_argument("--audio", type=Path, default=AUDIO_DIR, help="ingest dir (metadata)")
    parser.add_argument("--out", type=Path, default=LESSONS_DIR, help="output directory")
    parser.add_argument("--force", action="store_true", help="redo existing episodes")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.clean.glob("*.json"))
    if args.episode:
        paths = [args.clean / f"{e}.json" for e in args.episode]

    done, skipped, failed = 0, 0, []
    for n, path in enumerate(paths, start=1):
        tag = f"[{n}/{len(paths)}] {path.stem}"
        meta = args.audio / f"{path.stem}.json"
        if not path.exists() or not meta.exists():
            log.error("%s is missing its clean transcript or episode metadata", tag)
            failed.append(path.stem)
            continue
        clean = CleanTranscript.model_validate_json(path.read_text(encoding="utf-8"))
        if not args.force and is_distilled(clean, args.out):
            skipped += 1
            continue
        episode = Episode.model_validate_json(meta.read_text(encoding="utf-8"))
        log.info("%s distilling %s", tag, episode.title)
        try:
            result = distill(clean, episode)
        except Exception as e:  # keep going; a rerun picks failures back up
            log.error("%s failed: %s", tag, e)
            failed.append(clean.episode_id)
            continue
        out = args.out / f"{clean.episode_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(out)
        done += 1
        log.info("%s done: %d lessons", tag, len(result.lessons))

    log.info("distilled %d, skipped %d existing, failed %d", done, skipped, len(failed))
    if failed:
        log.error("failed ids: %s", " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
