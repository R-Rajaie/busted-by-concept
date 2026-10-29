"""Re-encode episode audio to 16 kHz mono Opus, about a fifth of the download's size.

Whisper and pyannote both resample to 16 kHz mono anyway, so nothing they use is lost.
Uses the ffmpeg binary from the `pack` extra (`uv sync --extra pack`); PyAV's encoder loop
is about 3x slower.
"""

import subprocess
from pathlib import Path

SAMPLE_RATE = 16_000
BITRATE = "24k"  # plenty for speech at 16 kHz
COMPRESSION = "3"  # 5x faster than ffmpeg's default 10, about 10% bigger


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def shrink(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    subprocess.run(
        [
            ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "libopus",
            "-b:a",
            BITRATE,
            "-compression_level",
            COMPRESSION,
            "-f",
            "ogg",
            str(tmp),
        ],
        check=True,
        capture_output=True,
    )
    tmp.replace(dst)
