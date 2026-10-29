"""Map diarized speakers to hosts -> `data/speakers/`.

uv run python -m coach.speakers
uv run python -m coach.speakers --rebootstrap   # re-average the by-elimination voiceprint

Cheap (no audio is read), so every run redoes every diarized episode. If the by-elimination
host has no voiceprint yet, the run labels them by elimination, averages a voiceprint from the
clearest episodes, saves it, and then matches all episodes again with it.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from coach.config import AUDIO_DIR, HOSTS_DIR, SPEAKERS_DIR, TRANSCRIPTS_RAW_DIR
from coach.ingest.models import Episode
from coach.speakers import identify
from coach.speakers.models import SpeakerMap, load_config, load_voiceprints
from coach.transcribe.diarize import EMBEDDING_MODEL
from coach.transcribe.models import DiarizedSpeaker, RawTranscript

log = logging.getLogger("coach.speakers")


def diarized(raw_dir: Path) -> dict[str, list[DiarizedSpeaker]]:
    out = {}
    for path in sorted(raw_dir.glob("*.json")):
        raw = RawTranscript.model_validate_json(path.read_text(encoding="utf-8"))
        if raw.diarized:
            out[raw.episode_id] = raw.speakers
    return out


def title(episode_id: str, audio_dir: Path) -> str:
    meta = audio_dir / f"{episode_id}.json"
    if not meta.exists():
        return ""
    return Episode.model_validate_json(meta.read_text(encoding="utf-8")).title


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.speakers", description=__doc__)
    parser.add_argument("--raw", type=Path, default=TRANSCRIPTS_RAW_DIR, help="input directory")
    parser.add_argument("--audio", type=Path, default=AUDIO_DIR, help="ingest dir (titles)")
    parser.add_argument("--out", type=Path, default=SPEAKERS_DIR, help="output directory")
    parser.add_argument(
        "--rebootstrap", action="store_true", help="discard the bootstrapped voiceprint first"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config()
    elim = config.by_elimination
    if args.rebootstrap and elim:
        (HOSTS_DIR / elim / "voiceprint.json").unlink(missing_ok=True)
    voiceprints = load_voiceprints(config)
    for host, vp in voiceprints.items():
        if vp.model != EMBEDDING_MODEL:
            log.error("%s's voiceprint uses %s, not %s; re-enroll", host, vp.model, EMBEDDING_MODEL)
            return 1
    if not voiceprints:
        log.error("no voiceprints in %s; run coach.speakers.enroll first", HOSTS_DIR)
        return 1

    speakers = diarized(args.raw)
    if not speakers:
        log.error("no diarized transcripts in %s", args.raw)
        return 1
    titles = {e: title(e, args.audio) for e in speakers}

    def run() -> list[SpeakerMap]:
        return [
            identify.identify(e, titles[e], s, voiceprints, config) for e, s in speakers.items()
        ]

    maps = run()
    if elim and elim not in voiceprints:
        vp = identify.bootstrap(elim, maps, speakers, config)
        if vp is None:
            log.warning(
                "too few clear episodes to average %s's voiceprint; keeping elimination labels",
                elim,
            )
        else:
            path = HOSTS_DIR / elim / "voiceprint.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(vp.model_dump_json(indent=2), encoding="utf-8")
            log.info("averaged %s's voiceprint from %d episodes -> %s", elim, vp.chunks, path)
            voiceprints[elim] = vp
            maps = run()

    args.out.mkdir(parents=True, exist_ok=True)
    for m in maps:
        (args.out / f"{m.episode_id}.json").write_text(
            m.model_dump_json(indent=2), encoding="utf-8"
        )

    seconds: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    for m in maps:
        for s in m.speakers:
            seconds[s.host] += s.seconds
            methods[s.method] += 1
    total = sum(seconds.values()) or 1
    log.info("%d episodes, %d with guests", len(maps), sum(m.guest_episode for m in maps))
    for host, s in seconds.most_common():
        log.info("  %-14s %5.1f h  %4.1f%%", host, s / 3600, 100 * s / total)
    log.info("  methods: %s", dict(methods))
    return 0


if __name__ == "__main__":
    sys.exit(main())
