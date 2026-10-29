"""Run the GPU transcription on Kaggle's free T4s, and babysit it from the laptop.

uv run python -m coach.cloud.kaggle prepare   # shrink audio, stage the two private datasets
                                              # (--limit 2 first, for a cheap trial run)
uv run python -m coach.cloud.kaggle upload    # create or update them on Kaggle
uv run python -m coach.cloud.kaggle run       # batches until every episode is done
uv run python -m coach.cloud.kaggle status

Needs the `pack` and `diarize` extras, `HF_TOKEN` (to copy the gated pyannote models into the
private dataset, so Kaggle never sees the token), and a Kaggle API token in
`~/.kaggle/access_token`. Each batch is one kernel run of up to 11 hours on two T4s; `run`
pushes a batch with every episode still missing, waits, downloads what finished, and repeats.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from coach import env
from coach.cloud.__main__ import CLOUD_DIR, add_code, pending, shrink_all
from coach.config import TRANSCRIPTS_RAW_DIR
from coach.transcribe.diarize import EMBEDDING_MODEL, PIPELINE
from coach.transcribe.models import RawTranscript

log = logging.getLogger("coach.cloud.kaggle")

KAGGLE_DIR = CLOUD_DIR / "kaggle"
AUDIO_SLUG = "bbc-audio"
CODE_SLUG = "bbc-code"
KERNEL_SLUG = "bbc-transcribe"
ACCELERATOR = "NvidiaTeslaT4"  # "GPU T4 x2"
GATED_MODELS = [PIPELINE, "pyannote/segmentation-3.0", EMBEDDING_MODEL]
POLL_S = 300
JOB = Path(__file__).with_name("kaggle_job.py")


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    return subprocess.run(
        [exe, *args], check=check, capture_output=True, text=True, encoding="utf-8"
    )


def username() -> str:
    m = re.search(r"username: (\S+)", kaggle("config", "view").stdout)
    if not m or m.group(1) == "None":
        sys.exit("no Kaggle username; put your API token in ~/.kaggle/access_token")
    return m.group(1)


def write_metadata(folder: Path, slug: str, title: str) -> None:
    meta = {"title": title, "id": f"{username()}/{slug}", "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def link(src: Path, dst: Path) -> None:
    """Hard-link to avoid a second copy of the audio; copy if linking isn't possible."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_hf_cache(hub: Path, token: str) -> None:
    """Lay the gated models out as a Hugging Face cache (`hub/models--org--name/refs/main`
    plus `snapshots/<commit>/`) with plain files, since Windows can't make the symlinks
    huggingface_hub normally uses. Offline lookups only read refs and snapshots."""
    from huggingface_hub import HfApi, snapshot_download

    for repo in GATED_MODELS:
        commit = HfApi().model_info(repo, token=token).sha
        root = hub / f"models--{repo.replace('/', '--')}"
        snapshot_download(repo, revision=commit, local_dir=root / "snapshots" / commit, token=token)
        shutil.rmtree(root / "snapshots" / commit / ".cache", ignore_errors=True)
        (root / "refs").mkdir(parents=True, exist_ok=True)
        (root / "refs" / "main").write_text(commit, encoding="utf-8")


def stage_code() -> Path:
    code = KAGGLE_DIR / "code"
    code.mkdir(parents=True, exist_ok=True)
    with tarfile.open(code / "code.tar", "w") as tar:
        add_code(tar)
    write_metadata(code, CODE_SLUG, "bbc code")
    return code


def prepare(args: argparse.Namespace) -> int:
    episodes = pending()
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        log.info("every episode is already transcribed and diarized")
        return 0
    small = shrink_all(episodes, CLOUD_DIR / "audio", args.workers)

    audio = KAGGLE_DIR / "audio"
    if audio.exists():
        shutil.rmtree(audio)  # only links and generated files, so rebuilding is cheap
    audio.mkdir(parents=True)
    for e in small:
        for name in (e.audio_file, f"{e.episode_id}.json"):
            link(CLOUD_DIR / "audio" / name, audio / name)

    cache = KAGGLE_DIR / "hf_build"
    token = env.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set; it's needed to copy the gated pyannote models")
    shutil.rmtree(cache, ignore_errors=True)
    build_hf_cache(cache / "hub", token)
    with tarfile.open(audio / "hf.tar", "w") as tar:
        tar.add(cache / "hub", arcname="hub")
    write_metadata(audio, AUDIO_SLUG, "bbc audio")

    stage_code()
    size = sum(f.stat().st_size for f in audio.iterdir()) / 1e9
    log.info("staged %d episodes (%.1f GB) in %s; next: upload", len(small), size, audio)
    return 0


def push_dataset(folder: Path, slug: str) -> None:
    ref = f"{username()}/{slug}"
    exists = kaggle("datasets", "status", ref, check=False).returncode == 0
    log.info(
        "%s %s (this can take a while for the audio)", "updating" if exists else "creating", ref
    )
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-t", "-q")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-t", "-q")
    while "ready" not in kaggle("datasets", "status", ref, check=False).stdout:
        time.sleep(15)
    log.info("%s is ready", ref)


def upload(args: argparse.Namespace) -> int:
    if not args.code_only:
        push_dataset(KAGGLE_DIR / "audio", AUDIO_SLUG)
    push_dataset(stage_code(), CODE_SLUG)  # always the current code
    return 0


def push_batch(episodes: list[str]) -> None:
    user = username()
    folder = KAGGLE_DIR / "kernel"
    folder.mkdir(parents=True, exist_ok=True)
    job = JOB.read_text(encoding="utf-8").replace(
        "EPISODES: list[str] = []", f"EPISODES: list[str] = {json.dumps(episodes)}", 1
    )
    (folder / "run.py").write_text(job, encoding="utf-8")
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # pip installs and the whisper model
        "dataset_sources": [f"{user}/{AUDIO_SLUG}", f"{user}/{CODE_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    out = kaggle("kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR)
    log.info("pushed a batch of %d episodes: %s", len(episodes), out.stdout.strip())


def kernel_status() -> str:
    """ "running", "queued", "complete", "error", "cancel..." or the raw text."""
    out = kaggle("kernels", "status", f"{username()}/{KERNEL_SLUG}", check=False)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?(\w+)"', out.stdout + out.stderr)
    return m.group(1).lower() if m else (out.stdout + out.stderr).strip()


def fetch(batch: int) -> int:
    """Download a finished run's outputs and copy valid transcripts in; returns the count."""
    dest = KAGGLE_DIR / "out" / f"batch{batch}"
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{username()}/{KERNEL_SLUG}", "-p", str(dest), "-o", "-q")
    TRANSCRIPTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(dest.rglob("*.json")):
        try:
            raw = RawTranscript.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            continue  # not a transcript
        if raw.partial or not raw.diarized:
            continue
        shutil.copy2(path, TRANSCRIPTS_RAW_DIR / f"{raw.episode_id}.json")
        count += 1
    return count


def staged_pending() -> list[str]:
    """Episodes still to do that are in the staged (and so uploaded) audio dataset."""
    return [
        e.episode_id for e in pending() if (KAGGLE_DIR / "audio" / f"{e.episode_id}.opus").exists()
    ]


def run(args: argparse.Namespace) -> int:
    for batch in range(1, args.max_batches + 1):
        todo = staged_pending()
        if not todo:
            log.info("every uploaded episode is transcribed and diarized")
            return 0
        push_batch(todo)
        time.sleep(60)
        while (status := kernel_status()) in ("queued", "running"):
            log.info("batch %d: %s", batch, status)
            time.sleep(POLL_S)
        got = fetch(batch)
        out = KAGGLE_DIR / "out" / f"batch{batch}"
        log.info("batch %d ended (%s): %d new transcripts, logs in %s", batch, status, got, out)
        if got == 0:
            log.error("no progress; check the logs above (an expired GPU quota or setup error?)")
            return 1
    log.info("stopped after %d batches; %d episodes left", args.max_batches, len(staged_pending()))
    return 0


def status(args: argparse.Namespace) -> int:
    log.info("kernel: %s", kernel_status())
    log.info("episodes still to transcribe: %d", len(pending()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.cloud.kaggle", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="shrink audio and stage the datasets")
    p.add_argument("--workers", type=int, default=2, help="parallel ffmpeg processes")
    p.add_argument("--limit", type=int, help="only the first N episodes (for a trial run)")
    u = sub.add_parser("upload", help="create or update the private datasets")
    u.add_argument("--code-only", action="store_true", help="only re-upload the code")
    r = sub.add_parser("run", help="push batches until every episode is done")
    r.add_argument("--max-batches", type=int, default=3, help="stop after this many")
    sub.add_parser("status", help="the kernel's state and what's left")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    commands = {"prepare": prepare, "upload": upload, "run": run, "status": status}
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
