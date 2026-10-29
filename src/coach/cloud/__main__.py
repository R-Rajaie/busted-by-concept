"""Pack the GPU run's inputs into one file, and unpack its results.

uv run python -m coach.cloud pack               # -> data/cloud/bundle.tar
uv run python -m coach.cloud unpack results.tar  # -> data/transcripts/raw/

The bundle holds the repo's tracked files plus shrunk audio as `data/audio/`, so on the GPU
box it unpacks into a working checkout. See docs/cloud.md for the whole run.
"""

import argparse
import logging
import re
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coach.cloud.shrink import shrink
from coach.config import AUDIO_DIR, DATA_DIR, ROOT, TRANSCRIPTS_RAW_DIR
from coach.ingest.models import Episode
from coach.transcribe.__main__ import ingested_episodes, is_transcribed

log = logging.getLogger("coach.cloud")

CLOUD_DIR = DATA_DIR / "cloud"
RESULT = re.compile(r"data/transcripts/raw/[\w-]{11}\.json")


def shrink_all(episodes: list[Episode], out_dir: Path, workers: int) -> list[Episode]:
    """Shrunk copies of each episode with matching metadata; existing ones are reused."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def one(n_episode: tuple[int, Episode]) -> Episode:
        n, e = n_episode
        small = e.model_copy(update={"audio_file": f"{e.episode_id}.opus"})
        if not small.audio_path(out_dir).exists():
            shrink(e.audio_path(AUDIO_DIR), small.audio_path(out_dir))
            log.info("[%d/%d] %s shrunk", n, len(episodes), e.episode_id)
        (out_dir / f"{e.episode_id}.json").write_text(
            small.model_dump_json(indent=2), encoding="utf-8"
        )
        return small

    with ThreadPoolExecutor(workers) as pool:  # each worker runs its own ffmpeg process
        small = list(pool.map(one, enumerate(episodes, start=1)))
    log.info("shrunk %d episodes in %.0f min", len(small), (time.monotonic() - started) / 60)
    return small


def add_code(tar: tarfile.TarFile) -> None:
    """The repo's files at their repo paths: tracked plus untracked-but-not-ignored, so
    uncommitted edits go along and secrets (`.env`, cookies) don't."""
    files = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for f in filter(None, files):
        if (ROOT / f).is_file():  # skip files deleted but not yet committed
            tar.add(ROOT / f, arcname=f, recursive=False)


def pending() -> list[Episode]:
    """Ingested episodes without a finished, diarized transcript."""
    return [
        e
        for e in ingested_episodes(AUDIO_DIR)
        if not is_transcribed(e.episode_id, TRANSCRIPTS_RAW_DIR, need_diarized=True)
    ]


def pack(args: argparse.Namespace) -> int:
    episodes = pending()
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        log.info("every episode is already transcribed and diarized")
        return 0
    audio_dir = CLOUD_DIR / "audio"
    small = shrink_all(episodes, audio_dir, args.workers)

    bundle = CLOUD_DIR / "bundle.tar"
    with tarfile.open(bundle, "w") as tar:
        add_code(tar)
        for e in small:
            for name in (e.audio_file, f"{e.episode_id}.json"):
                tar.add(audio_dir / name, arcname=f"data/audio/{name}")
    size = bundle.stat().st_size / 1e9
    log.info("wrote %s: %d episodes, %.1f GB", bundle, len(small), size)
    return 0


def unpack(args: argparse.Namespace) -> int:
    TRANSCRIPTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(args.results) as tar:
        for member in tar:
            if not member.isfile():
                continue
            if RESULT.fullmatch(member.name):
                dest = TRANSCRIPTS_RAW_DIR / Path(member.name).name
            elif member.name.endswith(".log") and "/" not in member.name:
                dest = CLOUD_DIR / member.name
            else:
                log.warning("ignoring %s", member.name)
                continue
            dest.write_bytes(tar.extractfile(member).read())
            count += dest.parent == TRANSCRIPTS_RAW_DIR
    log.info("unpacked %d transcripts into %s", count, TRANSCRIPTS_RAW_DIR)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.cloud", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack", help="shrink audio and bundle it with the code")
    p.add_argument("--workers", type=int, default=2, help="parallel ffmpeg processes")
    p.add_argument("--limit", type=int, help="only the first N episodes (for a trial run)")
    u = sub.add_parser("unpack", help="copy transcripts out of the GPU box's results.tar")
    u.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return pack(args) if args.command == "pack" else unpack(args)


if __name__ == "__main__":
    sys.exit(main())
