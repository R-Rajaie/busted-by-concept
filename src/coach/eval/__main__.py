"""Run the evals and compare with the previous run.

uv run python -m coach.eval

Results go to `data/eval/<timestamp>.json`. Run before and after any retrieval or prompt
change, and report both numbers.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from coach import glossary
from coach.config import DATA_DIR, TRANSCRIPTS_CLEAN_DIR, TRANSCRIPTS_RAW_DIR
from coach.correct.models import CleanTranscript
from coach.eval import metrics
from coach.index.search import search
from coach.transcribe.models import RawTranscript

RESULTS_DIR = DATA_DIR / "eval"


def retrieval(questions: list[metrics.Question]) -> dict:
    rows = []
    for q in questions:
        rank = metrics.first_hit_rank(metrics.citations(search(q.question)), q.expected)
        rows.append({"question": q.question, "rank": rank, "verified": q.verified})
        print(f"  {'hit @' + str(rank) if rank else 'MISS  '}  {q.question}")
    n = len(rows) or 1
    return {
        "hit_rate": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "questions": rows,
    }


def _window_text(segments, start: float, end: float) -> str:
    return " ".join(s.text for s in segments if s.end >= start and s.start <= end)


def term_error_rate(refs: list[metrics.GlossaryRef]) -> dict:
    gloss = glossary.load()
    totals = {"raw": [0, 0], "clean": [0, 0]}  # [terms, missed]
    rows = []
    for ref in refs:
        raw_path = TRANSCRIPTS_RAW_DIR / f"{ref.episode_id}.json"
        clean_path = TRANSCRIPTS_CLEAN_DIR / f"{ref.episode_id}.json"
        if not raw_path.exists() or not clean_path.exists():
            print(f"  skipping {ref.episode_id}@{ref.start:.0f}: transcript missing")
            continue
        raw = RawTranscript.model_validate_json(raw_path.read_text(encoding="utf-8"))
        clean = CleanTranscript.model_validate_json(clean_path.read_text(encoding="utf-8"))
        row = {"episode_id": ref.episode_id, "start": ref.start, "verified": ref.verified}
        for name, segments in (("raw", raw.segments), ("clean", clean.segments)):
            hyp = _window_text(segments, ref.start, ref.end)
            n, missed = metrics.term_errors(ref.text, hyp, gloss)
            totals[name][0] += n
            totals[name][1] += len(missed)
            row[f"{name}_missed"] = missed
        rows.append(row)
        print(
            f"  {ref.episode_id}@{ref.start:.0f}: raw missed {row['raw_missed'] or 'none'}, "
            f"clean missed {row['clean_missed'] or 'none'}"
        )
    rate = {k: (m / t if t else 0.0) for k, (t, m) in totals.items()}
    return {
        "raw_ter": rate["raw"],
        "clean_ter": rate["clean"],
        "terms": totals["raw"][0],
        "refs": rows,
    }


def previous(results_dir: Path) -> dict | None:
    runs = sorted(results_dir.glob("*.json"))
    return json.loads(runs[-1].read_text(encoding="utf-8")) if runs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.eval", description=__doc__)
    parser.add_argument("--label", default="", help="note stored with the results")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    questions, refs = metrics.load_questions(), metrics.load_refs()
    unverified = sum(not q.verified for q in questions) + sum(not r.verified for r in refs)
    if unverified:
        print(f"note: {unverified} eval items are unverified seeds; label them by hand\n")

    print(f"Retrieval ({len(questions)} questions)")
    ret = retrieval(questions)
    print(f"\nGlossary term error rate ({len(refs)} references)")
    ter = term_error_rate(refs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prev = previous(RESULTS_DIR)
    result = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "retrieval": ret,
        "glossary": ter,
    }
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    def line(name: str, now: float, before: float | None) -> str:
        delta = f"  (was {before:.0%})" if before is not None else ""
        return f"  {name:<22} {now:.0%}{delta}"

    p_ret = prev["retrieval"] if prev else {}
    p_ter = prev["glossary"] if prev else {}
    print("\nSummary")
    print(line("retrieval hit rate", ret["hit_rate"], p_ret.get("hit_rate")))
    print(line("retrieval MRR", ret["mrr"], p_ret.get("mrr")))
    print(line("term error, raw", ter["raw_ter"], p_ter.get("raw_ter")))
    print(line("term error, clean", ter["clean_ter"], p_ter.get("clean_ter")))
    print("  speaker attribution    skipped (no diarization yet)")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
