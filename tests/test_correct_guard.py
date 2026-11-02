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
