"""Project-wide paths and source settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLeEqhVGXNVw1jQqEO30jRteBdH6qLKPhS"
TRANSCRIPTS_RAW_DIR = DATA_DIR / "transcripts" / "raw"
TRANSCRIPTS_CLEAN_DIR = DATA_DIR / "transcripts" / "clean"
LESSONS_DIR = DATA_DIR / "lessons"
CONCEPTS_DIR = DATA_DIR / "concepts"
INDEX_DIR = DATA_DIR / "index"
SPEAKERS_DIR = DATA_DIR / "speakers"

GLOSSARY_DIR = ROOT / "glossary"
HOSTS_DIR = ROOT / "hosts"

DDRAGON_URL = "https://ddragon.leagueoflegends.com"
