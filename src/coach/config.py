"""Project-wide paths and source settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLeEqhVGXNVw1jQqEO30jRteBdH6qLKPhS"
