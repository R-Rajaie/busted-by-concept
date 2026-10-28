from datetime import date

from coach.correct.llm_pass import change_ratio
from coach.correct.matcher import Matcher
from coach.distill.__main__ import season_for
from coach.glossary import Champion, Glossary, Item, SlangTerm
from coach.ingest.models import Episode
from coach.transcribe import hallucination, hotwords
from coach.transcribe.models import Segment

GLOSSARY = Glossary(
    patch="test",
    champions=[Champion(id="Shaco", name="Shaco"), Champion(id="Vi", name="Vi")],
    items=[Item(id="1", name="Rabadon's Deathcap")],
    slang=[
        SlangTerm(term="win con", misheard=["wing con"], core=True),
        SlangTerm(term="side laning", misheard=["side landing"], core=True),
    ],
)


def seg(i: int, text: str) -> Segment:
    return Segment(id=i, start=i, end=i + 1, text=text, avg_logprob=0, no_speech_prob=0)


def test_repeated_run_is_flagged():
    segments = [seg(0, "Real talk."), *(seg(i, "Thanks for watching!") for i in range(1, 4))]
    assert hallucination.flag(segments) == 3
    assert not segments[0].suspect


def test_loop_within_segment_is_flagged():
    segments = [seg(0, "you know what " * 8)]
    assert hallucination.flag(segments) == 1


def test_slang_keeps_sentence_case():
    m = Matcher(GLOSSARY)
    assert m.apply_slang("Side landing and wing con.") == "Side laning and win con."


def test_fuzzy_applies_only_confident_matches():
    m = Matcher(GLOSSARY)
    text, hints = m.apply_fuzzy("A Shacco jungler")
    assert text == "A Shaco jungler"
    assert hints == []
    # Short names are excluded entirely, so ordinary words never become "Vi".
    assert m.apply_fuzzy("We went in")[0] == "We went in"


def test_change_ratio():
    assert change_ratio(["abcd"], ["abcd"]) == 0
    assert change_ratio(["abcd"], ["abce"]) == 0.25


def test_hotwords_put_episode_champions_first():
    episode = Episode(episode_id="x", url="u", title="Shaco jungle guide", audio_file="x.webm")
    words = hotwords.build(episode, GLOSSARY)
    assert words[0] == "Shaco"
    assert "win con" in words


def test_season_mapping():
    assert season_for(date(2024, 6, 1)) == ("S14", None)
    assert season_for(None)[0] is None
    assert season_for(date(2025, 1, 3))[0] is None  # too close to the boundary to guess
