import numpy as np

from coach.correct.models import CleanSegment
from coach.distill.__main__ import attribute
from coach.speakers import identify
from coach.speakers.enroll import average, video_id
from coach.speakers.models import HostConfig, Voiceprint
from coach.transcribe.diarize import Turn, assign_speakers
from coach.transcribe.models import DiarizedSpeaker, Segment, Word

CONFIG = HostConfig(hosts=["curtis", "nathan", "charlie"], by_elimination="charlie")
DIM = 8


def axis(i: int, noise: float = 0.0) -> list[float]:
    v = np.zeros(DIM)
    v[i] = 1.0
    v[(i + 1) % DIM] = noise
    return list(v)


def voiceprint(host: str, i: int) -> Voiceprint:
    return Voiceprint(
        host=host, model="m", source="enrolled", embedding=axis(i), from_ids=[], chunks=1
    )


VOICEPRINTS = {"curtis": voiceprint("curtis", 0), "nathan": voiceprint("nathan", 1)}


def speaker(label: str, i: int, seconds: float = 600) -> DiarizedSpeaker:
    return DiarizedSpeaker(label=label, seconds=seconds, embedding=axis(i, noise=0.2))


def by_label(speakers, title="Ep 1 - Wave Management", voiceprints=VOICEPRINTS):
    m = identify.identify("ep", title, speakers, voiceprints, CONFIG)
    return {s.label: (s.host, s.method) for s in m.speakers}


def test_leftover_speaker_is_charlie_by_elimination():
    got = by_label([speaker("A", 0), speaker("B", 1), speaker("C", 2), speaker("D", 3, 30)])
    assert got == {
        "A": ("curtis", "voiceprint"),
        "B": ("nathan", "voiceprint"),
        "C": ("charlie", "elimination"),
        "D": ("uncertain", "short"),
    }


def test_no_elimination_in_guest_episodes_or_with_two_leftovers():
    speakers = [speaker("A", 0), speaker("B", 1), speaker("C", 2)]
    assert by_label(speakers, title="Ep 35 ft. Zaboutine")["C"] == ("guest_unknown", "no_match")
    two = by_label([speaker("A", 0), speaker("C", 2), speaker("E", 4)])
    assert two["C"] == two["E"] == ("guest_unknown", "no_match")


def test_bootstrapped_voiceprint_rejects_a_stranger():
    maps, speakers = [], {}
    for n in range(identify.MIN_BOOTSTRAP):
        s = [speaker("A", 0), speaker("B", 1), speaker("C", 2)]
        speakers[f"ep{n}"] = s
        maps.append(identify.identify(f"ep{n}", "", s, VOICEPRINTS, CONFIG))
    vp = identify.bootstrap("charlie", maps, speakers, CONFIG)
    assert vp is not None and vp.chunks == identify.MIN_BOOTSTRAP
    with_charlie = VOICEPRINTS | {"charlie": vp}
    got = by_label([speaker("C", 2), speaker("X", 5)], voiceprints=with_charlie)
    assert got == {"C": ("charlie", "voiceprint"), "X": ("guest_unknown", "no_match")}


def test_too_few_episodes_means_no_bootstrap():
    s = [speaker("A", 0), speaker("B", 1), speaker("C", 2)]
    maps = [identify.identify("ep", "", s, VOICEPRINTS, CONFIG)]
    assert identify.bootstrap("charlie", maps, {"ep": s}, CONFIG) is None


def word(start: float, end: float) -> Word:
    return Word(start=start, end=end, text="x", prob=1.0)


def segment(i: int, words: list[Word]) -> Segment:
    return Segment(
        id=i,
        start=words[0].start,
        end=words[-1].end,
        text="",
        avg_logprob=0,
        no_speech_prob=0,
        words=words,
    )


def test_assign_speakers_by_overlap_and_majority():
    segs = [segment(0, [word(0, 1), word(1, 2), word(2, 3)]), segment(1, [word(50, 51)])]
    turns = [Turn(0, 2.2, "S0"), Turn(2.2, 10, "S1"), Turn(100, 101, "S0")]
    assign_speakers(segs, turns)
    assert [w.speaker for w in segs[0].words] == ["S0", "S0", "S1"]
    assert segs[0].speaker == "S0"
    assert segs[1].words[0].speaker is None and segs[1].speaker is None


def test_attribute_needs_one_dominant_host():
    words = [word(t, t + 1) for t in range(10)]
    for w in words:
        w.speaker = "S0"
    words[9].speaker = "S1"
    seg = CleanSegment(**segment(0, words).model_dump(), raw_text="")
    hosts = {"S0": "curtis", "S1": "nathan"}
    assert attribute(0, 10, [seg], hosts) == "curtis"
    assert attribute(7, 10, [seg], hosts) == "uncertain"  # 2 of 3 words is under MIN_SHARE
    assert attribute(0, 10, [seg], {}) == "uncertain"  # no speaker map yet


def test_enroll_drops_chunks_unlike_the_rest():
    rows = np.array([identify.unit(axis(0, 0.1)) for _ in range(9)] + [identify.unit(axis(4))])
    vp, keep = average(rows)
    assert keep.sum() == 9 and vp @ identify.unit(axis(0)) > 0.99


def test_video_id_from_urls():
    assert video_id("https://www.youtube.com/watch?v=Dn2l7Kg8bb4&t=3") == "Dn2l7Kg8bb4"
    assert video_id("https://youtu.be/qq5_nuL2Q5U") == "qq5_nuL2Q5U"


def test_kaggle_job_has_the_batch_placeholder():
    from coach.cloud import kaggle

    assert "EPISODES: list[str] = []" in kaggle.JOB.read_text(encoding="utf-8")
