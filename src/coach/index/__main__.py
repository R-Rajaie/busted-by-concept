"""Build the search index -> `data/index/`.

uv run python -m coach.index --rebuild
"""

import argparse
import logging
import sys
from pathlib import Path

from coach.config import CONCEPTS_DIR, INDEX_DIR, TRANSCRIPTS_CLEAN_DIR
from coach.consolidate.models import Consolidated
from coach.correct.models import CleanTranscript
from coach.index.build import build_docs, write_table

log = logging.getLogger("coach.index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.index", description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        required=True,
        help="rebuild the whole index (the only mode for now)",
    )
    parser.add_argument("--concepts", type=Path, default=CONCEPTS_DIR, help="consolidate output")
    parser.add_argument("--clean", type=Path, default=TRANSCRIPTS_CLEAN_DIR, help="transcripts")
    parser.add_argument("--out", type=Path, default=INDEX_DIR, help="output directory")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = args.concepts / "consolidated.json"
    if not path.exists():
        log.error("%s not found; run coach.consolidate first", path)
        return 1
    consolidated = Consolidated.model_validate_json(path.read_text(encoding="utf-8"))
    transcripts = [
        CleanTranscript.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(args.clean.glob("*.json"))
    ]
    log.info(
        "%d concepts, %d lessons, %d transcripts",
        len(consolidated.concepts),
        len(consolidated.lessons),
        len(transcripts),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(args.out, build_docs(consolidated, transcripts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
