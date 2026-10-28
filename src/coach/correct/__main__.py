"""Correct proper nouns and jargon in raw transcripts -> `data/transcripts/clean/`.

uv run python -m coach.correct --all
uv run python -m coach.correct --episode <id> --no-llm   # deterministic matcher only
"""

import argparse
import logging
import sys
from pathlib import Path

from coach import glossary
from coach.config import TRANSCRIPTS_CLEAN_DIR, TRANSCRIPTS_RAW_DIR
from coach.correct import llm_pass
from coach.correct.matcher import Match, Matcher
from coach.correct.models import CleanSegment, CleanTranscript, Correction, RejectedChunk
from coach.transcribe.models import RawTranscript

log = logging.getLogger("coach.correct")


def is_corrected(raw: RawTranscript, out_dir: Path) -> bool:
    path = out_dir / f"{raw.episode_id}.json"
    if not path.exists():
        return False
    clean = CleanTranscript.model_validate_json(path.read_text(encoding="utf-8"))
    return clean.partial == raw.partial  # a finished raw transcript supersedes a partial one


def correct(raw: RawTranscript, matcher: Matcher, use_llm: bool) -> CleanTranscript:
    segments = [
        CleanSegment(**s.model_dump(), raw_text=s.text) for s in raw.segments if not s.suspect
    ]
    corrections: list[Correction] = []
    hints: dict[int, list[Match]] = {}

    for s in segments:
        text = matcher.apply_slang(s.text)
        if text != s.text:
            corrections.append(
                Correction(segment_id=s.id, before=s.text, after=text, source="slang")
            )
        fuzzed, hints[s.id] = matcher.apply_fuzzy(text)
        if fuzzed != text:
            corrections.append(
                Correction(segment_id=s.id, before=text, after=fuzzed, source="fuzzy")
            )
        s.text = fuzzed

    rejected: list[RejectedChunk] = []
    if use_llm:
        context = llm_pass.glossary_context(matcher.glossary, raw.hotwords)
        step = llm_pass.CHUNK_SEGMENTS
        for c in range(0, len(segments), step):
            chunk = segments[c : c + step]
            fixes = llm_pass.correct_chunk(
                [(s.id, s.text) for s in chunk],
                [h for s in chunk for h in hints[s.id]],
                context,
            )
            before = [s.text for s in chunk]
            after = [fixes.get(s.id, s.text) for s in chunk]
            ratio = llm_pass.change_ratio(before, after)
            if ratio > llm_pass.MAX_CHANGE:
                log.warning(
                    "  rejected LLM chunk %d-%d: %.1f%% changed",
                    chunk[0].id,
                    chunk[-1].id,
                    ratio * 100,
                )
                rejected.append(
                    RejectedChunk(
                        first_segment=chunk[0].id, last_segment=chunk[-1].id, change_ratio=ratio
                    )
                )
                continue
            for s, new in zip(chunk, after, strict=True):
                if new != s.text:
                    corrections.append(
                        Correction(segment_id=s.id, before=s.text, after=new, source="llm")
                    )
                    s.text = new
            log.info("  chunk %d/%d", c // step + 1, -(-len(segments) // step))

    return CleanTranscript(
        episode_id=raw.episode_id,
        partial=raw.partial,
        segments=segments,
        corrections=corrections,
        rejected_chunks=rejected,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.correct", description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--all", action="store_true", help="every raw transcript")
    target.add_argument("--episode", action="append", metavar="ID", help="episode id (repeatable)")
    parser.add_argument("--raw", type=Path, default=TRANSCRIPTS_RAW_DIR, help="input directory")
    parser.add_argument("--out", type=Path, default=TRANSCRIPTS_CLEAN_DIR, help="output directory")
    parser.add_argument("--force", action="store_true", help="redo existing episodes")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM pass")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    matcher = Matcher(glossary.load())

    paths = sorted(args.raw.glob("*.json"))
    if args.episode:
        paths = [args.raw / f"{e}.json" for e in args.episode]

    done, skipped, failed = 0, 0, []
    for n, path in enumerate(paths, start=1):
        tag = f"[{n}/{len(paths)}] {path.stem}"
        if not path.exists():
            log.error("%s has no raw transcript", tag)
            failed.append(path.stem)
            continue
        raw = RawTranscript.model_validate_json(path.read_text(encoding="utf-8"))
        if not args.force and is_corrected(raw, args.out):
            skipped += 1
            continue
        log.info("%s correcting %d segments", tag, len(raw.segments))
        try:
            clean = correct(raw, matcher, use_llm=not args.no_llm)
        except Exception as e:  # keep going; a rerun picks failures back up
            log.error("%s failed: %s", tag, e)
            failed.append(raw.episode_id)
            continue
        out = args.out / f"{raw.episode_id}.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(clean.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(out)
        done += 1
        log.info(
            "%s done: %d corrections, %d rejected chunks",
            tag,
            len(clean.corrections),
            len(clean.rejected_chunks),
        )

    log.info("corrected %d, skipped %d existing, failed %d", done, skipped, len(failed))
    if failed:
        log.error("failed ids: %s", " ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
