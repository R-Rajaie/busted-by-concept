from coach.eval import metrics
from coach.glossary import Glossary, SlangTerm
from coach.serve import ddragon, riot


def participant(pid: int, puuid: str, team: int, champ: str, pos: str, **kw) -> dict:
    return {
        "participantId": pid,
        "puuid": puuid,
        "teamId": team,
        "championName": champ,
        "teamPosition": pos,
        "win": team == 100,
        "kills": 3,
        "deaths": 2,
        "assists": 5,
        "totalMinionsKilled": 150,
        "neutralMinionsKilled": 10,
        "visionScore": 20,
    } | kw


MATCH = {
    "metadata": {"matchId": "NA1_1"},
    "info": {
        "gameDuration": 1800,
        "participants": [
            participant(1, "me", 100, "Ahri", "MIDDLE"),
            participant(6, "them", 200, "Zed", "MIDDLE"),
        ],
    },
}


def frame(ts: int, events: list[dict], cs: int = 0) -> dict:
    return {
        "timestamp": ts,
        "events": events,
        "participantFrames": {"1": {"minionsKilled": cs, "jungleMinionsKilled": 2}},
    }


def test_summarize_reads_early_game_from_timeline():
    death = {"type": "CHAMPION_KILL", "victimId": 1, "timestamp": 5 * 60_000}
    late_death = {"type": "CHAMPION_KILL", "victimId": 1, "timestamp": 20 * 60_000}
    frames = [frame(i * 60_000, [], cs=i * 8) for i in range(12)]
    frames[5]["events"].append(death)
    frames[11]["events"].append(late_death)
    g = riot.summarize(MATCH, {"info": {"frames": frames}}, "me")
    assert (g.champion, g.role, g.lane_opponent, g.win) == ("Ahri", "middle", "Zed", True)
    assert g.cs == 160 and g.cs_at_10 == 82 and g.deaths_before_14 == 1


def test_format_history_keeps_record_without_timeline():
    g = riot.summarize(MATCH, None, "me")
    assert g.cs_at_10 is None
    assert riot.format_history([g]).startswith("1W 0L over 1 games; avg vision 20")


def doc(episode: str, start: float, end: float) -> dict:
    return {"episode_id": episode, "start_ts": start, "end_ts": end}


def test_first_hit_rank_uses_window_with_slack():
    expected = [metrics.Expected(episode_id="a", start=300, end=400)]
    cited = [doc("b", 300, 400), doc("a", 0, 100), doc("a", 420, 500)]
    assert metrics.first_hit_rank(cited, expected) == 3
    assert metrics.first_hit_rank(cited[:2], expected) is None


def test_term_errors_ignore_hyphens_and_case():
    gloss = Glossary(patch="t", champions=[], items=[], slang=[SlangTerm(term="win con")])
    assert metrics.term_errors("the win con is", "the Win-con is", gloss) == (1, [])
    assert metrics.term_errors("the win con is", "the wing con is", gloss) == (1, ["win con"])


def test_find_champion_id_handles_punctuation():
    assert ddragon.find_champion_id("kaisa") == "Kaisa"
    assert ddragon.find_champion_id("Kai'Sa") == "Kaisa"
    assert ddragon.find_champion_id("not a champ") is None
