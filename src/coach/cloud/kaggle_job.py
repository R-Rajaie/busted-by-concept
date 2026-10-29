"""The Kaggle side of `coach.cloud.kaggle`: transcribe and diarize one batch on two T4s.

Pushed as a script kernel with the batch's episode ids filled in. Standard library only,
since it runs in Kaggle's own Python before the project's environment exists. Outputs go to
/kaggle/working/raw/, which `coach.cloud.kaggle` downloads when the run finishes.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

EPISODES: list[str] = []  # filled in by coach.cloud.kaggle
MODEL = "large-v3-turbo"  # several times faster than large-v3 on a T4, nearly as accurate
DEADLINE_S = 11 * 3600  # Kaggle stops sessions at 12 h; stop early so the outputs get saved
GPUS = 2

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "raw"
CODE = Path("/tmp/coach")
HF_HOME = Path("/tmp/hf")
started = time.monotonic()


def log(msg: str) -> None:
    print(f"[{(time.monotonic() - started) / 60:6.1f} min] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> str:
    log("+ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    done = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if done.returncode:
        print(done.stdout[-5000:], done.stderr[-5000:], sep="\n", flush=True)
        sys.exit(f"command failed with exit code {done.returncode}")
    return done.stdout


def first(pattern: str) -> Path | None:
    return next(iter(sorted(INPUT.rglob(pattern))), None)


def unpack(tar_name: str, marker: str, dest: Path) -> None:
    """Kaggle may or may not extract an uploaded tar, so accept the tar or its contents
    (found by `marker`, a path inside it)."""
    dest.mkdir(parents=True, exist_ok=True)
    tar = first(tar_name)
    if tar is not None:
        with tarfile.open(tar) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python without extraction filters; our own tars, so fine
                t.extractall(dest)
        return
    found = first(marker)
    if found is None:
        sys.exit(f"neither {tar_name} nor {marker} under {INPUT}; are both datasets attached?")
    root = found.parents[len(Path(marker).parts) - 1]
    shutil.copytree(root, dest, dirs_exist_ok=True)


def setup() -> tuple[Path, dict[str, str]]:
    unpack("code.tar", "src/coach/config.py", CODE)
    unpack("hf.tar", "hub/models--pyannote--segmentation-3.0", HF_HOME)
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh([sys.executable, "-m", "uv", "sync", "--frozen", "--extra", "diarize"], cwd=CODE)
    python = str(CODE / ".venv" / "bin" / "python")
    libs = sh(
        [
            python,
            "-c",
            "import nvidia.cublas.lib as b, nvidia.cudnn.lib as d;"
            "print(b.__path__[0] + ':' + d.__path__[0])",
        ]
    ).strip()
    env = os.environ | {
        "HF_HOME": str(HF_HOME),
        # pyannote keeps its own cache (~/.cache/torch/pyannote) rather than HF_HOME's.
        "PYANNOTE_CACHE": str(HF_HOME / "hub"),
        "LD_LIBRARY_PATH": f"{libs}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PYTHONUNBUFFERED": "1",
    }
    # The whisper model isn't gated, so fetch it now; then go offline so the gated pyannote
    # models load from the uploaded cache without a token.
    sh(
        [python, "-c", f"from faster_whisper import download_model; download_model({MODEL!r})"],
        env=env,
    )
    audio = first("*.opus")
    if audio is None:
        sys.exit(f"no audio under {INPUT}")
    return audio.parent, env | {"HF_HUB_OFFLINE": "1"}


def main() -> None:
    if not EPISODES:
        sys.exit("no episodes in this batch")
    audio_dir, env = setup()
    OUT.mkdir(parents=True, exist_ok=True)
    python = str(CODE / ".venv" / "bin" / "python")
    procs = []
    for gpu in range(GPUS):
        shard = EPISODES[gpu::GPUS]
        if not shard:
            continue
        args = [python, "-m", "coach.transcribe", "--audio", str(audio_dir), "--out", str(OUT)]
        args += ["--device", "cuda", "--diarize", "--model", MODEL]
        for e in shard:
            args += ["--episode", e]
        logfile = open(WORK / f"gpu{gpu}.log", "w")  # closed when the session ends
        procs.append(
            subprocess.Popen(
                args,
                cwd=CODE,
                env=env | {"CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )
        )
        log(f"GPU {gpu}: {len(shard)} episodes")

    while any(p.poll() is None for p in procs):
        time.sleep(60)
        done = len(list(OUT.glob("*.json")))
        log(f"{done}/{len(EPISODES)} episodes done")
        if time.monotonic() - started > DEADLINE_S:
            log("deadline reached; stopping so the finished episodes are saved")
            for p in procs:
                p.terminate()
            for p in procs:
                p.wait(timeout=120)
            break

    for tmp in OUT.glob("*.tmp"):
        tmp.unlink()  # an episode cut off mid-write
    log(f"finished: {len(list(OUT.glob('*.json')))}/{len(EPISODES)} episodes")


if __name__ == "__main__":
    main()
