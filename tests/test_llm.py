from datetime import datetime

from coach.llm import LIMIT, _seconds_until_reset

MSG = "You've hit your session limit · resets 1am (America/Los_Angeles)"


def test_limit_message_waits_until_the_reset():
    assert LIMIT.search(MSG)
    wait = _seconds_until_reset(MSG, now=datetime(2026, 11, 1, 23, 50))
    assert wait == 70 * 60 + 120  # 11:50 pm -> 1:00 am, plus slack
    later = _seconds_until_reset("resets 2:30pm", now=datetime(2026, 11, 2, 9, 0))
    assert later == 5.5 * 3600 + 120


def test_limit_without_a_time_has_no_wait():
    assert _seconds_until_reset("You've hit your weekly limit · resets Nov 5") is None
