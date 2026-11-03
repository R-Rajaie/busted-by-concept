from coach.correct.llm_pass import segment_ok

VOCAB = {"kayle", "league", "tp", "hardstuck", "jungle", "broken", "by", "tank", "bruiser", "gank"}


def test_keeps_word_forms_of_known_terms():
    assert segment_ok("he's really tinky", "he's really tanky", VOCAB)
    assert segment_ok("a bit bruisy-er", "a bit bruisery", VOCAB)
    assert segment_ok("she got gang", "she got ganked", VOCAB)
    assert segment_ok("tankyer than", "tankier than", VOCAB)
    assert segment_ok("very gang-able", "very gankable", VOCAB)
    assert segment_ok("in Kale's lane", "in Kayle's lane", VOCAB)
    for tp in ("TPed", "TP'd", "TPd", "TP'ed", "TPing", "TP'ing", "TPs"):
        assert segment_ok("he teepeed bot", f"he {tp} bot", VOCAB), tp
    assert not segment_ok("we have gotten better", "we have got better", VOCAB)


def test_keeps_name_and_jargon_fixes():
    assert segment_ok("I played Kale all game", "I played Kayle all game", VOCAB)
    assert segment_ok("two teepees up", "two TPs up", VOCAB)
    assert segment_ok("I was heart's stuck", "I was hardstuck", VOCAB)
    assert segment_ok("he was out-jongling me", "he was out-jungling me", VOCAB)
    assert segment_ok("welcome to Brokeback Concept", "welcome to Broken By Concept", VOCAB)
    assert segment_ok("Kale.", "Kale", VOCAB)  # punctuation only


def test_blocks_rewording_and_moved_text():
    assert not segment_ok("welcome to Broken by Concept", "welcome to Busted by Concept", VOCAB)
    assert not segment_ok("we have gotten better", "we have got better", VOCAB)
    moved = "Anything you would add there? I love playing a hard game"
    assert not segment_ok(moved, "I love playing a hard game", VOCAB)  # sentence deleted
    assert not segment_ok("and", "and I love playing a hard game like League", VOCAB)


def test_reapply_takes_blocked_edits_that_now_pass(tmp_path):
    from coach.correct.__main__ import reapply
    from coach.correct.models import CleanSegment, CleanTranscript, Correction, RejectedChunk
    from coach.glossary import Glossary

    def seg(i: int, text: str) -> CleanSegment:
        return CleanSegment(
            id=i, start=i, end=i + 1, text=text, raw_text=text, avg_logprob=0, no_speech_prob=0
        )

    clean = CleanTranscript(
        episode_id="ep",
        partial=False,
        segments=[seg(0, "kill the grump"), seg(1, "he was a liar"), seg(2, "grump")],
        corrections=[Correction(segment_id=2, before="grimp", after="grump", source="slang")],
        rejected_chunks=[
            RejectedChunk(
                first_segment=0, last_segment=0, change_ratio=0, proposed="kill the Gromp"
            ),
            RejectedChunk(
                first_segment=1, last_segment=1, change_ratio=0, proposed="he was a Lion"
            ),
            RejectedChunk(first_segment=2, last_segment=2, change_ratio=0, proposed="Gromp"),
        ],
    )
    (tmp_path / "ep.json").write_text(clean.model_dump_json(), encoding="utf-8")

    class M:
        glossary = Glossary(patch="", champions=[], items=[], slang=[])

    reapply(tmp_path, None, M())
    got = CleanTranscript.model_validate_json((tmp_path / "ep.json").read_text(encoding="utf-8"))
    assert [s.text for s in got.segments] == ["kill the Gromp", "he was a liar", "grump"]
    assert [r.first_segment for r in got.rejected_chunks] == [1, 2]  # unknown word; reslanged
