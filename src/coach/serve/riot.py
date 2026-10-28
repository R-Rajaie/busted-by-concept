"""Riot API client for match history, with rate limiting and an on-disk match cache.

Development keys expire every 24 hours and allow 20 requests/second and 100 per 2 minutes.
Matches and timelines never change once played, so they're cached under `data/cache/riot/`.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from coach import env
from coach.config import DATA_DIR

log = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache" / "riot"
MIN_INTERVAL_S = 1.25  # 100 requests / 120 s
EARLY_GAME_MS = 14 * 60 * 1000

# Platform (where the account plays) -> regional routing for account-v1 and match-v5.
REGIONS = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "me1": "europe",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
    "sg2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}


class RiotError(RuntimeError):
    pass


class Riot:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or env.get("RIOT_API_KEY")
        if not self.api_key:
            raise RiotError("RIOT_API_KEY is not set; add it to .env")
        self._last = 0.0

    def _get(self, region: str, path: str, retries: int = 3) -> Any:
        url = f"https://{region}.api.riotgames.com{path}"
        for _ in range(retries):
            wait = self._last + MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            req = urllib.request.Request(url, headers={"X-Riot-Token": self.api_key})
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    delay = float(e.headers.get("Retry-After", 10))
                    log.warning("rate limited; sleeping %.0fs", delay)
                    time.sleep(delay)
                    continue
                if e.code in (401, 403):
                    raise RiotError("Riot API key rejected (dev keys expire every 24h)") from e
                if e.code == 404:
                    raise RiotError(f"not found: {path}") from e
                raise RiotError(f"Riot API error {e.code} for {path}") from e
        raise RiotError(f"still rate limited after {retries} tries: {path}")

    def _cached(self, kind: str, match_id: str, region: str, path: str) -> Any:
        file = CACHE_DIR / kind / f"{match_id}.json"
        if file.exists():
            return json.loads(file.read_text(encoding="utf-8"))
        data = self._get(region, path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps(data), encoding="utf-8")
        return data

    def puuid(self, riot_id: str, region: str) -> str:
        name, _, tag = riot_id.partition("#")
        if not tag:
            raise RiotError("use a Riot ID like Name#TAG")
        path = (
            f"/riot/account/v1/accounts/by-riot-id/{urllib.parse.quote(name)}/"
            f"{urllib.parse.quote(tag)}"
        )
        return self._get(region, path)["puuid"]

    def match_ids(self, puuid: str, region: str, count: int, queue: int | None) -> list[str]:
        query = f"?start=0&count={count}" + (f"&queue={queue}" if queue else "")
        return self._get(region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids{query}")

    def match(self, match_id: str, region: str) -> dict:
        return self._cached("match", match_id, region, f"/lol/match/v5/matches/{match_id}")

    def timeline(self, match_id: str, region: str) -> dict:
        return self._cached(
            "timeline", match_id, region, f"/lol/match/v5/matches/{match_id}/timeline"
        )


@dataclass
class GameSummary:
    match_id: str
    champion: str
    role: str
    win: bool
    duration_min: float
    kills: int
    deaths: int
    assists: int
    cs: int
    cs_at_10: int | None
    deaths_before_14: int | None
    vision_score: int
    lane_opponent: str | None


def summarize(match: dict, timeline: dict | None, puuid: str) -> GameSummary:
    info = match["info"]
    me = next(p for p in info["participants"] if p["puuid"] == puuid)
    role = me.get("teamPosition") or me.get("individualPosition") or ""
    opponent = next(
        (
            p["championName"]
            for p in info["participants"]
            if p["teamId"] != me["teamId"] and role and p.get("teamPosition") == role
        ),
        None,
    )
    cs_at_10 = early_deaths = None
    if timeline:
        pid = me["participantId"]
        frames = timeline["info"]["frames"]
        if len(frames) > 10:
            f = frames[10]["participantFrames"][str(pid)]
            cs_at_10 = f["minionsKilled"] + f["jungleMinionsKilled"]
        early_deaths = sum(
            1
            for frame in frames
            for ev in frame["events"]
            if ev["type"] == "CHAMPION_KILL"
            and ev.get("victimId") == pid
            and ev["timestamp"] < EARLY_GAME_MS
        )
    return GameSummary(
        match_id=match["metadata"]["matchId"],
        champion=me["championName"],
        role=role.lower(),
        win=me["win"],
        duration_min=round(info["gameDuration"] / 60, 1),
        kills=me["kills"],
        deaths=me["deaths"],
        assists=me["assists"],
        cs=me["totalMinionsKilled"] + me["neutralMinionsKilled"],
        cs_at_10=cs_at_10,
        deaths_before_14=early_deaths,
        vision_score=me["visionScore"],
        lane_opponent=opponent,
    )


def match_history(riot_id: str, platform: str = "na1", count: int = 10) -> list[GameSummary]:
    """Recent ranked solo/duo games (queue 420) with early-game stats from timelines."""
    region = REGIONS.get(platform.lower())
    if region is None:
        raise RiotError(f"unknown platform {platform!r}; one of {', '.join(REGIONS)}")
    riot = Riot()
    puuid = riot.puuid(riot_id, region)
    games = []
    for mid in riot.match_ids(puuid, region, count, queue=420):
        games.append(summarize(riot.match(mid, region), riot.timeline(mid, region), puuid))
    return games


def format_history(games: list[GameSummary]) -> str:
    if not games:
        return "No recent ranked solo/duo games."
    rows = []
    for g in games:
        rows.append(
            f"{'W' if g.win else 'L'} {g.champion} {g.role} vs {g.lane_opponent or '?'}, "
            f"{g.kills}/{g.deaths}/{g.assists}, {g.cs} CS in {g.duration_min} min "
            f"(CS@10 {g.cs_at_10}), deaths before 14:00: {g.deaths_before_14}, "
            f"vision {g.vision_score}"
        )
    n, wins = len(games), sum(g.win for g in games)
    cs10 = [g.cs_at_10 for g in games if g.cs_at_10 is not None]
    early = [g.deaths_before_14 for g in games if g.deaths_before_14 is not None]
    parts = [f"{wins}W {n - wins}L over {n} games"]
    if cs10:
        parts.append(f"avg CS@10 {sum(cs10) / len(cs10):.0f}")
    if early:
        parts.append(f"avg deaths before 14:00 {sum(early) / len(early):.1f}")
    parts.append(f"avg vision {sum(g.vision_score for g in games) / n:.0f}")
    return "; ".join(parts) + "\n" + "\n".join(rows)
