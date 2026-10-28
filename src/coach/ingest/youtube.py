"""Playlist listing and audio download via yt-dlp."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL

from coach.ingest.models import Episode

log = logging.getLogger(__name__)

# Placeholder titles yt-dlp returns for entries we can't download.
UNAVAILABLE_TITLES = {"[Private video]", "[Deleted video]"}


def _base_opts(cookies: Path | None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        # YouTube's player JS needs a runtime to solve its challenges; use what's installed.
        "js_runtimes": {"deno": {}, "node": {}, "bun": {}},
    }
    if cookies:
        opts["cookiefile"] = str(cookies)
    return opts


def list_playlist(url: str, cookies: Path | None = None) -> list[tuple[int, str, str]]:
    """Return `(playlist_index, video_id, title)` for every available entry, in playlist order."""
    opts = _base_opts(cookies) | {"extract_flat": "in_playlist"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = []
    for i, entry in enumerate(info.get("entries") or [], start=1):
        if not entry or not entry.get("id"):
            continue
        title = entry.get("title") or ""
        if title in UNAVAILABLE_TITLES:
            log.warning("skipping unavailable entry %s (%s)", entry["id"], title)
            continue
        entries.append((i, entry["id"], title))
    return entries


def download_episode(
    video_id: str,
    audio_dir: Path,
    playlist_index: int | None = None,
    cookies: Path | None = None,
) -> Episode:
    """Download the best audio-only stream as-is (no re-encode) and return its metadata."""
    opts = _base_opts(cookies) | {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "overwrites": True,
        "continuedl": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        audio_path = Path(ydl.prepare_filename(info))

    upload_date = info.get("upload_date")
    return Episode(
        episode_id=video_id,
        url=info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        title=info.get("title") or "",
        description=info.get("description") or "",
        upload_date=datetime.strptime(upload_date, "%Y%m%d").date() if upload_date else None,
        duration_s=info.get("duration"),
        playlist_index=playlist_index,
        audio_file=audio_path.name,
    )
