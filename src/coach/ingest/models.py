"""On-disk records written by the ingest stage."""

from datetime import date
from pathlib import Path

from pydantic import BaseModel


class Episode(BaseModel):
    """Metadata for one downloaded episode, stored as `data/audio/<episode_id>.json`.

    The JSON file is written only after the audio finishes downloading, so its
    presence marks the episode as ingested.
    """

    episode_id: str
    source: str = "youtube"
    url: str
    title: str
    description: str = ""
    upload_date: date | None = None
    duration_s: float | None = None
    playlist_index: int | None = None
    audio_file: str

    def audio_path(self, audio_dir: Path) -> Path:
        return audio_dir / self.audio_file
