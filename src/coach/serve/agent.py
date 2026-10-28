"""The coach: a tool-calling loop over podcast search, champion data, and match history.

Each step the model returns either one tool call or the final answer, as structured JSON.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from coach import llm
from coach.index.search import format_result, search
from coach.serve import ddragon, riot

log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 6
TOOL_OUTPUT_CHARS = 8000

SYSTEM = """You are a League of Legends improvement coach. Your coaching philosophy comes from
the Broken by Concept podcast, which you reach through the search_podcast tool. Live game
data is the ground truth for numbers.

Tools:
- search_podcast(query): concept cards with the podcast's current stance, cited lessons, and
  transcript excerpts. Search before giving any advice; search again with other wording if
  the first results miss.
- get_champion_data(champion): current-patch stats, abilities, cooldowns, and costs.
- get_match_history(riot_id, platform, count): the player's recent ranked games with CS@10,
  deaths before 14:00, and vision score.

Rules:
- Cite the episode title and timestamp link for every claim drawn from the podcast.
- Use live data for numbers (stats, item costs, cooldowns), never podcast quotes.
- When reviewing match history, find patterns first (CS@10, early deaths, vision), then
  search for lessons that target those patterns.
- Lead with a concept card's current stance and use its lessons as supporting evidence.
- Never present superseded advice as current. Only discuss how a view changed if asked,
  using the concept card's history.
- When hosts disagree, present each view with attribution. Speakers marked "uncertain" are
  not identified: say "the hosts" and never guess who said it.
- Say when advice comes from an older season or is patch-dependent (durability "patch").
- If the podcast doesn't cover something, say so rather than inventing an answer.
- Never invent host credentials.

Reply with one action per step: a tool call, or "answer" with your final reply to the
player in markdown."""


class Step(BaseModel):
    action: Literal["search_podcast", "get_champion_data", "get_match_history", "answer"]
    query: str | None = Field(None, description="search_podcast: what to search for")
    champion: str | None = Field(None, description="get_champion_data: champion name")
    riot_id: str | None = Field(None, description="get_match_history: Name#TAG")
    platform: str | None = Field(None, description="get_match_history: e.g. na1, euw1, kr")
    count: int | None = Field(None, description="get_match_history: games to fetch (max 20)")
    answer: str | None = Field(None, description="answer: the final reply")


@dataclass
class Coach:
    riot_id: str | None = None
    platform: str = "na1"
    history: list[tuple[str, str]] = field(default_factory=list)  # (question, answer)

    def _run_tool(self, step: Step) -> str:
        try:
            if step.action == "search_podcast":
                return format_result(search(step.query or "")) or "No results."
            if step.action == "get_champion_data":
                return ddragon.champion_summary(step.champion or "")
            if step.action == "get_match_history":
                riot_id = step.riot_id or self.riot_id
                if not riot_id:
                    return "No Riot ID given. Ask the player for their Name#TAG."
                games = riot.match_history(
                    riot_id, step.platform or self.platform, min(step.count or 10, 20)
                )
                return riot.format_history(games)
        except Exception as e:  # tool errors go back to the model, which can explain them
            return f"Tool error: {e}"
        return f"Unknown action {step.action}"

    def _context(self) -> str:
        lines = [f"Today is {date.today()}."]
        try:
            lines.append(f"Current patch: {ddragon.latest_patch()}.")
        except Exception:
            pass
        if self.riot_id:
            lines.append(f"The player's Riot ID is {self.riot_id} on {self.platform}.")
        for q, a in self.history[-3:]:
            lines.append(f"\nEarlier question: {q}\nYour earlier answer: {a}")
        return "\n".join(lines)

    def ask(self, question: str) -> str:
        transcript = [self._context(), f"\nPlayer: {question}"]
        for n in range(MAX_TOOL_CALLS + 1):
            if n == MAX_TOOL_CALLS:
                transcript.append("\nTool budget used up. Answer now with what you have.")
            step = llm.complete_model("\n".join(transcript), SYSTEM, Step, model=llm.SMART_MODEL)
            if step.action == "answer" or n == MAX_TOOL_CALLS:
                answer = step.answer or "I couldn't put an answer together."
                self.history.append((question, answer))
                return answer
            args = step.model_dump(exclude_none=True, exclude={"action", "answer"})
            log.info("tool %s %s", step.action, args)
            output = self._run_tool(step)[:TOOL_OUTPUT_CHARS]
            transcript.append(f"\nTool call: {step.action} {args}\nResult:\n{output}")
        raise AssertionError("unreachable")
