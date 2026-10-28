"""Ask the coach a question, or start an interactive session.

uv run python -m coach.serve "how do I stop dying early as jungle?"
uv run python -m coach.serve --riot-id Name#TAG --platform na1     # interactive
"""

import argparse
import logging
import sys

from coach.serve.agent import Coach


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.serve", description=__doc__)
    parser.add_argument("question", nargs="?", help="one question; omit for interactive mode")
    parser.add_argument("--riot-id", help="your Riot ID, Name#TAG, for match history")
    parser.add_argument("--platform", default="na1", help="na1, euw1, kr, ...")
    parser.add_argument("-v", "--verbose", action="store_true", help="log tool calls")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    coach = Coach(riot_id=args.riot_id, platform=args.platform)
    if args.question:
        print(coach.ask(args.question))
        return 0

    print("Ask the coach anything. Ctrl+C or an empty line to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print("\n" + coach.ask(question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
