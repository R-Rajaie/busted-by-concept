"""Build the LanceDB table that search runs over: concept cards, lessons, and raw chunks."""

import logging
from collections.abc import Iterable
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

from coach import embed
from coach.consolidate.models import Consolidated
from coach.correct.models import CleanSegment, CleanTranscript
from coach.distill.models import Lesson

log = logging.getLogger(__name__)

TABLE = "docs"
CHUNK_S = 75  # raw transcript chunk length
WRITE_BATCH = 1000  # docs embedded and written at a time


class Doc(LanceModel):
    id: str
    kind: str  # "concept" | "lesson" | "chunk"
    text: str
    vector: Vector(embed.EMBED_DIM)  # type: ignore[valid-type]
    episode_id: str = ""
    start_ts: float = 0
    end_ts: float = 0
    concept_id: str = ""
    label: str = ""
    status: str = ""
    season: str = ""
    durability: str = ""
    episode_date: str = ""
    speaker: str = ""
    recency_rank: float = 1.0  # 0 = oldest, 1 = newest within its concept


def recency_ranks(lessons: list[Lesson]) -> dict[str, float]:
    """Rank standing lessons by last_reaffirmed within each concept. Singletons get 1.0."""
    by_concept: dict[str, list[Lesson]] = {}
    for x in lessons:
        if x.concept_id and x.status != "superseded":
            by_concept.setdefault(x.concept_id, []).append(x)
    ranks: dict[str, float] = {}
    for members in by_concept.values():
        members.sort(key=lambda x: (x.last_reaffirmed or x.episode_date or x.id, x.start_ts))
        n = len(members)
        for i, x in enumerate(members):
            ranks[x.id] = 1.0 if n == 1 else i / (n - 1)
    return ranks


def chunks(transcript: CleanTranscript, size: float = CHUNK_S) -> list[list[CleanSegment]]:
    out: list[list[CleanSegment]] = []
    for s in transcript.segments:
        if not out or s.end - out[-1][0].start > size:
            out.append([])
        out[-1].append(s)
    return out


def build_docs(consolidated: Consolidated, transcripts: Iterable[CleanTranscript]) -> list[dict]:
    """Docs without vectors; `write_table` embeds them. Pass transcripts as a generator so
    only one (with its word timings) is in memory at a time."""
    lessons = {x.id: x for x in consolidated.lessons}
    ranks = recency_ranks(consolidated.lessons)
    docs: list[dict] = []

    for c in consolidated.concepts:
        text = f"{c.label}\n{c.stance}"
        if c.history:
            text += f"\nHistory: {c.history}"
        docs.append(
            dict(
                id=c.id,
                kind="concept",
                text=text,
                concept_id=c.id,
                label=c.label,
                episode_date=str(c.last_reaffirmed or ""),
            )
        )

    for x in lessons.values():
        docs.append(
            dict(
                id=x.id,
                kind="lesson",
                text=f"{x.claim}\n{x.reasoning}",
                episode_id=x.episode_id,
                start_ts=x.start_ts,
                end_ts=x.end_ts,
                concept_id=x.concept_id or "",
                status=x.status or "",
                season=x.season or "",
                durability=x.durability,
                episode_date=str(x.episode_date or ""),
                speaker=x.speaker,
                recency_rank=ranks.get(x.id, 1.0),
            )
        )

    season_of = {
        x.episode_id: (x.season or "", str(x.episode_date or "")) for x in lessons.values()
    }
    for t in transcripts:
        season, day = season_of.get(t.episode_id, ("", ""))
        for n, chunk in enumerate(chunks(t)):
            docs.append(
                dict(
                    id=f"{t.episode_id}#{n:04d}",
                    kind="chunk",
                    text=" ".join(s.text for s in chunk),
                    episode_id=t.episode_id,
                    start_ts=chunk[0].start,
                    end_ts=chunk[-1].end,
                    season=season,
                    episode_date=day,
                )
            )

    return docs


def write_table(index_dir: Path, docs: list[dict], batch: int = WRITE_BATCH) -> None:
    """Embed and add the docs a batch at a time, so memory stays flat however many there are."""
    db = lancedb.connect(index_dir)
    table = db.create_table(TABLE, schema=Doc, mode="overwrite")
    for k in range(0, len(docs), batch):
        part = docs[k : k + batch]
        vecs = embed.embed([d["text"] for d in part])
        table.add(
            [Doc(**d, vector=v.tolist()).model_dump() for d, v in zip(part, vecs, strict=True)]
        )
        log.info("  embedded %d/%d", min(k + batch, len(docs)), len(docs))
    table.create_fts_index("text", replace=True)
    log.info("indexed %d docs", len(docs))
