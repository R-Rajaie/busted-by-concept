"""Hybrid search: BM25 + embeddings, reranked, then concept-first grouping.

uv run python -m coach.index.search "when should I freeze the wave"
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path

import lancedb

from coach import embed
from coach.config import AUDIO_DIR, INDEX_DIR
from coach.distill.__main__ import season_for
from coach.index.build import TABLE
from coach.ingest.models import Episode

CANDIDATES = 40  # per retriever, before reranking
# Patch staleness per season behind the current one. Philosophy drift is handled by
# status labels instead, so these only reflect how fast game facts go out of date.
DECAY = {"patch": 0.5, "macro": 0.85, "evergreen": 0.97}
UNKNOWN_SEASON_GAP = 1


@dataclass
class Hit:
    doc: dict
    relevance: float  # reranker score, 0-1
    score: float  # after recency and staleness weighting

    @property
    def id(self) -> str:
        return self.doc["id"]


@dataclass
class ConceptResult:
    concept: Hit
    evidence: list[Hit] = field(default_factory=list)


@dataclass
class SearchResult:
    concepts: list[ConceptResult]
    chunks: list[Hit]  # raw transcript, for quotes and context


@cache
def _table(index_dir: Path = INDEX_DIR):
    return lancedb.connect(index_dir).open_table(TABLE)


def _season_num(season: str) -> int | None:
    m = re.fullmatch(r"S(\d+)", season or "")
    return int(m.group(1)) if m else None


def staleness(doc: dict, today: date | None = None) -> float:
    current = _season_num(season_for(today or date.today())[0] or "")
    num = _season_num(doc["season"])
    gap = UNKNOWN_SEASON_GAP if num is None or current is None else max(0, current - num)
    return DECAY.get(doc["durability"] or "macro", DECAY["macro"]) ** gap


def weight(doc: dict, relevance: float) -> float:
    if doc["kind"] == "concept":
        return relevance  # a card's stance is already the current view
    score = relevance * staleness(doc)
    if doc["kind"] == "lesson":
        # Recency only matters within a concept; a lone old lesson gets rank 1.0.
        score *= 0.5 + 0.5 * doc["recency_rank"]
    return score


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _rerank(query: str, docs: list[dict]) -> list[Hit]:
    rel = embed.rerank(query, [d["text"] for d in docs])
    hits = [Hit(d, r, weight(d, r)) for d, r in zip(docs, rel, strict=True)]
    return sorted(hits, key=lambda h: h.score, reverse=True)


def search(
    query: str,
    n_concepts: int = 3,
    n_evidence: int = 3,
    n_chunks: int = 3,
    include_superseded: bool = False,
) -> SearchResult:
    table = _table()
    vec = embed.embed_query(query)
    found: dict[str, dict] = {}
    for d in table.search(vec).distance_type("cosine").limit(CANDIDATES).to_list():
        found[d["id"]] = d
    for d in table.search(query, query_type="fts").limit(CANDIDATES).to_list():
        found.setdefault(d["id"], d)
    docs = [d for d in found.values() if include_superseded or d["status"] != "superseded"]
    hits = _rerank(query, docs)

    # A concept is as relevant as its best card-or-lesson hit.
    concept_score: dict[str, float] = {}
    for h in hits:
        cid = h.doc["concept_id"]
        if cid and h.doc["kind"] in ("concept", "lesson"):
            concept_score[cid] = max(concept_score.get(cid, 0), h.score)
    top = sorted(concept_score, key=concept_score.get, reverse=True)[:n_concepts]

    concepts = []
    for cid in top:
        rows = table.search().where(f"concept_id = '{_escape(cid)}'").limit(500).to_list()
        card = next(r for r in rows if r["kind"] == "concept")
        lessons = [
            r
            for r in rows
            if r["kind"] == "lesson" and (include_superseded or r["status"] != "superseded")
        ]
        card_hit = next((h for h in hits if h.id == cid), None)
        if card_hit is None:
            card_hit = Hit(card, concept_score[cid], concept_score[cid])
        concepts.append(ConceptResult(card_hit, _rerank(query, lessons)[:n_evidence]))

    chunks = [h for h in hits if h.doc["kind"] == "chunk"][:n_chunks]
    return SearchResult(concepts, chunks)


def _ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


@cache
def _titles(audio_dir: Path = AUDIO_DIR) -> dict[str, str]:
    out = {}
    for meta in audio_dir.glob("*.json"):
        e = Episode.model_validate_json(meta.read_text(encoding="utf-8"))
        out[e.episode_id] = e.title
    return out


def cite(doc: dict) -> str:
    """'"Title" at 0:03:27 (https://youtu.be/<id>?t=207), 2026-10-26 S16'"""
    eid, start = doc["episode_id"], int(doc["start_ts"])
    title = _titles().get(eid, eid)
    when = " ".join(x for x in (doc["episode_date"], doc["season"]) if x)
    return f'"{title}" at {_ts(start)} (https://youtu.be/{eid}?t={start}), {when}'


def format_result(result: SearchResult) -> str:
    """Plain-text rendering used by the CLI and the agent's search tool."""
    out = []
    for c in result.concepts:
        d = c.concept.doc
        out.append(f"## {d['label']}  [{d['id']}, relevance {c.concept.relevance:.2f}]")
        out.append(d["text"].split("\n", 1)[1] if "\n" in d["text"] else d["text"])
        for e in c.evidence:
            x = e.doc
            status = f", {x['status']}" if x["status"] else ""
            out.append(
                f"  - {x['text'].splitlines()[0]}\n    [{cite(x)}, {x['durability']}"
                f"{status}, speaker {x['speaker']}]"
            )
        out.append("")
    if result.chunks:
        out.append("## Transcript excerpts")
        for h in result.chunks:
            out.append(f"  - [{cite(h.doc)}] {h.doc['text'][:400]}")
    return "\n".join(out).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coach.index.search", description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--superseded", action="store_true", help="include superseded lessons")
    args = parser.parse_args(argv)
    print(format_result(search(args.query, include_superseded=args.superseded)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
