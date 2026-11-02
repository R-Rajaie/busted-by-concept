"""Rank the words the correction guard blocked most often, to find terms missing from the
glossary. Each blocked segment records the text the LLM proposed; the words it brought in that
aren't known terms are counted, with an example.

    uv run python scripts/blocked_terms.py [--top 60]
"""

import argparse
import difflib
from collections import Counter

from coach import glossary
from coach.config import TRANSCRIPTS_CLEAN_DIR
from coach.correct.llm_pass import _known, _token, vocabulary
from coach.correct.models import CleanTranscript


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()

    vocab = vocabulary(glossary.load(), [])
    counts: Counter[str] = Counter()
    example: dict[str, str] = {}
    for path in sorted(TRANSCRIPTS_CLEAN_DIR.glob("*.json")):
        clean = CleanTranscript.model_validate_json(path.read_text(encoding="utf-8"))
        text = {s.id: s.raw_text for s in clean.segments}
        for r in clean.rejected_chunks:
            if not r.proposed:
                continue
            a, b = text.get(r.first_segment, "").split(), r.proposed.split()
            ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
            for op, i1, i2, j1, j2 in ops:
                if op == "equal" or max(i2 - i1, j2 - j1) > 3:
                    continue  # long rewrites are blocked on purpose, not for a missing word
                for t in map(_token, b[j1:j2]):
                    if t and not _known(t, vocab):
                        counts[t] += 1
                        example.setdefault(t, f"{' '.join(a[i1:i2])!r} -> {' '.join(b[j1:j2])!r}")
    for word, n in counts.most_common(args.top):
        print(f"{n:4}  {word:<18} {example[word]}")


if __name__ == "__main__":
    main()
