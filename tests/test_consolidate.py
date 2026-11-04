from datetime import date

import numpy as np

from coach import llm
from coach.consolidate import __main__ as consolidate
from coach.consolidate.models import Consolidated
from coach.consolidate.review import Review
from coach.distill.models import EpisodeLessons, Lesson


def lesson(n: int) -> Lesson:
    return Lesson(
        claim=f"claim {n}",
        reasoning="",
        role="any",
        phase="lane",
        champions=[],
        durability="evergreen",
        start_ts=n,
        end_ts=n + 1,
        id=f"ep-{n}",
        episode_id="ep",
        episode_date=date(2024, 1, 1),
        season=None,
        speaker="curtis",
    )


def test_a_stopped_run_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    lessons_dir, out = tmp_path / "lessons", tmp_path / "concepts"
    lessons_dir.mkdir()
    ep = EpisodeLessons(episode_id="ep", partial=False, lessons=[lesson(n) for n in range(4)])
    (lessons_dir / "ep.json").write_text(ep.model_dump_json(), encoding="utf-8")
    # four lessons pointing four different ways, so four concepts
    monkeypatch.setattr(consolidate.embed, "embed", lambda texts: np.eye(8)[: len(texts)])

    calls = []

    def review(members):
        calls.append(members[0].id)
        if len(calls) == 3:
            raise llm.LLMStop("weekly limit")
        return Review(label=members[0].claim, misfit_ids=[], statuses=[], stance="", history="")

    monkeypatch.setattr(consolidate, "review", review)
    args = ["--all", "--lessons", str(lessons_dir), "--out", str(out)]
    assert consolidate.main(args) == 1
    assert (out / "progress.json").exists()

    assert consolidate.main(["--resume", "--out", str(out)]) == 0
    assert calls == ["ep-0", "ep-1", "ep-2", "ep-2", "ep-3"]  # only the stopped one again
    assert not (out / "progress.json").exists()
    got = Consolidated.model_validate_json((out / "consolidated.json").read_text("utf-8"))
    assert [c.label for c in got.concepts] == [f"claim {n}" for n in range(4)]
