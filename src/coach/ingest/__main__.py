"""Download podcast episodes to `data/audio/`.

uv run python -m coach.ingest --all
uv run python -m coach.ingest --episode <video_id>
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

from coach.config import AUDIO_DIR, PLAYLIST_URL
from coach.ingest.models import Episode
from coach.ingest.youtube import download_episode, list_playlist

log = logging.getLogger("coach.ingest")


def is_ingested(episode_id: str, audio_dir: Path) -> bool:
    meta = audio_dir / f"{episode_id}.json"
    if not meta.exists():
        return False
    episode = Episode.model_validate_json(meta.read_text(encoding="utf-8"))
    return episode.audio_path(audio_dir).exists()


def write_episode(episode: Episode, audio_dir: Path) -> None:
    meta = audio_dir / f"{episode.episode_id}.json"
    tmp = meta.with_suffix(".json.tmp")
    tmp.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(meta)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.ingest", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every episode in the playlist")
    target.add_argument("--episode", action="append", metavar="ID", help="video id (repeatable)")
    parser.add_argument("--playlist", default=PLAYLIST_URL, help="playlist URL")
    parser.add_argument("--out", type=Path, default=AUDIO_DIR, help="output directory")
    parser.add_argument("--force", action="store_true", help="re-download existing episodes")
    parser.add_argument("--limit", type=int, help="stop after this many new downloads")
    parser.add_argument("--cookies", type=Path, help="Netscape cookies.txt if YouTube blocks you")
    parser.add_argument("--sleep", type=float, default=5.0, help="mean seconds between downloads")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.all:
        log.info("listing playlist %s", args.playlist)
        targets = list_playlist(args.playlist, cookies=args.cookies)
        log.info("%d available episodes", len(targets))
    else:
        targets = [(None, vid, "") for vid in args.episode]

    downloaded, skipped, failed = 0, 0, []
    for n, (index, video_id, title) in enumerate(targets, start=1):
        tag = f"[{n}/{len(targets)}] {video_id}"
        if not args.force and is_ingested(video_id, args.out):
            skipped += 1
            continue
        if args.limit is not None and downloaded >= args.limit:
            log.info("hit --limit %d, stopping", args.limit)
            break
        if downloaded:
            time.sleep(random.uniform(0.5, 1.5) * args.sleep)
        log.info("%s downloading %s", tag, title)
        try:
            episode = download_episode(video_id, args.out, index, cookies=args.cookies)
        except Exception as e:  # keep going; a rerun picks failures back up
            log.error("%s failed: %s", tag, e)
            failed.append(video_id)
            continue
        write_episode(episode, args.out)
        downloaded += 1
        log.info("%s done -> %s", tag, episode.audio_file)

    log.info("downloaded %d, skipped %d existing, failed %d", downloaded, skipped, len(failed))
    if failed:
        log.error("failed ids: %s", " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
