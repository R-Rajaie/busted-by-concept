# CLAUDE.md

## Project

A League of Legends improvement coach built on retrieval-augmented generation (RAG) over ~450 hours of transcribed episodes of <PODCAST_NAME>, combined with live game data from Data Dragon and the Riot API. This is NOT a fine-tuned model. The podcast is the coaching philosophy; live data is ground truth for numbers, items, and patch facts.

The repo is private. Podcast content is not ours to redistribute.

## Pipeline

Each stage reads the previous stage's output and writes its own. Stages are keyed by `episode_id` (for YouTube sources, the video id).

1. **ingest**: download audio via yt-dlp from the YouTube playlist in `src/coach/config.py` → `data/audio/<id>.<ext>` plus `data/audio/<id>.json` (`Episode` metadata; written last, so it marks completion)
2. **transcribe**: WhisperX large-v3 with VAD, diarization, word timestamps, and per-episode hotwords, then speaker identification against enrolled host voiceprints → `data/transcripts/raw/`. Currently implemented with faster-whisper (`distil-large-v3`, CPU int8, ~1.8x realtime) and no diarization, so every lesson's `speaker` is `uncertain` until diarization and speaker ID land. `--max-seconds N` writes a `partial` transcript for testing; downstream stages redo an episode when its partial flag changes.
3. **correct**: glossary fuzzy + phonetic matching, then an LLM proper-noun pass → `data/transcripts/clean/`
4. **distill**: an LLM extracts structured lessons as JSON → `data/lessons/`
5. **consolidate**: cluster lessons into concepts, label each lesson's status chronologically, and write concept cards → `data/concepts/`
6. **index**: hybrid search (BM25 + embeddings) with a reranker, over concept cards, lessons, and raw chunks → `data/index/`
7. **serve**: an agent loop with tools `search_podcast`, `get_champion_data`, `get_match_history`
8. **eval**: retrieval hit rate and glossary term error rate

## Layout

```
src/coach/
  config.py     # paths, playlist URL
  ingest/       # audio download, episode metadata
  transcribe/   # whisperx wrapper, hotword builder
  correct/      # fuzzy/phonetic matcher, LLM cleanup
  distill/      # lesson extraction, Lesson schema
  consolidate/  # concept clustering, status labeling, concept cards
  index/        # chunking, embeddings, BM25, reranker
  serve/        # agent, tools, system prompt
  eval/         # question sets, metrics
glossary/
  champions.json   # generated from Data Dragon, do not hand-edit
  items.json       # generated from Data Dragon, do not hand-edit
  slang.yaml       # hand-curated jargon and host catchphrases
hosts/
  <host_id>/
    reference/     # clean solo clips for voiceprint enrollment
    profile.json   # stated background claims, each with an episode citation
data/              # gitignored
tests/
```

## Commands

```
uv sync
uv run python -m coach.ingest --all                # --episode <id>, --limit N, --cookies cookies.txt
uv run python -m coach.transcribe --episode <id>   # or --all; --limit N, --max-seconds N, --model
uv run python -m coach.correct --all
uv run python -m coach.distill --all
uv run python -m coach.consolidate --all            # or --episode <id> to update only touched concepts
uv run python -m coach.index --rebuild
uv run python -m coach.index.search "query"        # inspect raw retrieval
uv run python -m coach.serve "question"            # or no question for interactive; --riot-id Name#TAG --platform na1
uv run python -m coach.eval --label "what changed" # saves data/eval/<time>.json, compares with the last run
uv run python -m coach.glossary.refresh            # re-pull Data Dragon
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Conventions

- Python 3.12, uv, ruff, type hints everywhere.
- Pydantic models for every on-disk record, especially `Lesson` and `Chunk`.
- Every stage is idempotent: skip an episode if its output already exists, unless `--force` is passed.
- Never modify an upstream stage's output in place. If a stage needs fixing, rerun it.
- Long runs (transcribe, distill) must be resumable and log progress per episode.
- Secrets (Riot API key, LLM API keys) live in `.env` and never get committed.

## Lesson schema

Each distilled lesson has: `claim`, `reasoning`, `role` (top/jungle/mid/adc/support/any), `phase` (draft/lane/mid/late/teamfight/mental), `champions` (list), `season` (e.g. "S14"), `episode_id`, `start_ts`, `end_ts`, `speaker`.

`speaker` is a host id from `hosts/`, `guest_unknown`, or `uncertain`.

Fields added by the consolidate stage: `concept_id`, `status` (reinforced/refined/superseded/evolved/disagreement), `superseded_by` (lesson id or null), `first_said`, `last_reaffirmed`.

Field added by the distill stage: `durability` (patch/macro/evergreen).

`season` is required. If an episode's date can't be mapped to a season, flag it rather than guessing.

## Consolidation and recency

The podcast's views evolve, so newer episodes should outweigh older ones, but only on the same concept. Never apply a global recency decay: it buries old concepts that were never revisited.

- **Concepts**: cluster lessons by embedding similarity, then have an LLM assign a canonical label (e.g. "wave management: when to freeze"). Clustering is imperfect, so spot-check clusters after big runs.
- **Status labels**: within each concept, an LLM reviews lessons in date order and labels each one:
  - `reinforced`: repeats an earlier point (raises confidence)
  - `refined`: adds nuance (both versions stay retrievable)
  - `superseded`: replaced by a later take (excluded from normal retrieval)
  - `evolved`: the same host changed their view over time (the later take leads, the history is kept)
  - `disagreement`: different hosts hold conflicting views in the same era (both are surfaced, attributed)
- **Concept cards**: each concept gets a short card with the current stance, a one-line history of how it changed, and a per-host stance section when hosts differ.
- **Ranking**: retrieve concept cards first, then their non-superseded lessons as evidence. Recency weighting applies only within a concept, e.g. `score = relevance * (0.5 + 0.5 * recency_rank_within_concept)`. A concept with a single old lesson gets no penalty.
- Use `last_reaffirmed`, not `first_said`, for recency. An old idea repeated recently counts as current.
- **Two separate kinds of aging**: philosophy drift is handled by status labels. Patch staleness is handled by `season` plus `durability`: `patch` lessons decay fast, `macro` moderately, and `evergreen` barely at all. Don't merge these into one decay.
- **Incremental updates**: when a new episode is added, assign its lessons to existing concepts (or create new ones) and rerun consolidation only for the concepts it touched.

## Glossary rules

- `champions.json` and `items.json` are regenerated from Data Dragon. Never edit them by hand.
- `slang.yaml` is hand-curated. Add to it when an eval or correction log shows a missed term, but do not bulk-generate entries.
- The hotword list is built per episode (core slang plus champions named in the title or description) because Whisper's prompt budget is about 224 tokens.

## Speaker attribution

- Diarization (pyannote via WhisperX) gives anonymous per-episode labels that reset each episode. Speaker identification maps them to hosts.
- Enroll each host from a few minutes of clean solo audio in `hosts/<host_id>/reference/`. Embed with pyannote's embedding model, SpeechBrain ECAPA-TDNN, or NeMo TitaNet (pick one and keep it consistent).
- Match each diarized speaker to the nearest host by cosine similarity. Below the threshold, label it `guest_unknown`, never a forced match.
- Use self-introductions ("I'm X") as a free check on the mapping.
- Segments under ~2 seconds, crosstalk, and short interjections are unreliable. Mark lessons from them `uncertain`.
- Misattribution invents fake disagreements, which is worse than no attribution. Before trusting a run, measure attribution error on the hand-labeled sample in `src/coach/eval/`.
- If a lesson attributed to host X contradicts X's stated position in the same episode, treat it as a likely diarization error and flag it instead of labeling it `evolved` or `disagreement`.
- Host background claims go in `profile.json` only when a host states them on air, each with an episode citation.

## Gotchas

- Whisper hallucinations: silence and intro music can produce repeated lines or invented text. Keep VAD on and run the repeated-line detector after transcription.
- The LLM correction pass must only fix proper nouns and jargon. Diff its output against its input and reject the chunk if more than ~5% of characters changed.
- Stale advice: retrieval down-weights older seasons, and the agent must say when advice comes from an older patch era.
- Riot API: development keys expire every 24 hours and are rate-limited. Respect the rate limits and cache match data.
- BM25 matters: embeddings handle champion and item names poorly, so don't drop the keyword half of hybrid search to simplify things.
- YouTube: yt-dlp needs a JS runtime (deno, node, or bun) on PATH for YouTube's player challenges. If YouTube starts bot-blocking, pass `--cookies` with an exported cookies.txt and raise `--sleep`.

## Agent behavior

- Cite episode and timestamp for every claim drawn from the podcast.
- Use live data for numbers (stats, item costs, cooldowns), never podcast quotes.
- When reviewing a player's match history, find patterns first (CS@10, early deaths, vision score), then retrieve lessons that target them.
- Lead with the concept card's current stance. Use older non-superseded lessons as supporting evidence.
- Never present superseded advice as current. Show it only when the user asks how a view has changed, and then use the concept card's history.
- When hosts disagree, present each host's view with attribution instead of picking one silently. When a single host's view evolved, lead with the latest and mention the change only if relevant.
- Describe a host's background only with claims from their `profile.json`, cited to an episode. Never invent credentials.

## Before changing retrieval or prompts

Run `coach.eval` before and after the change, and report both hit rates. Most quality problems are retrieval problems, so check retrieval before tweaking prompts.

## TODO / unknowns

- <PODCAST_NAME> (source is the YouTube playlist in `src/coach/config.py`)
- Transcription hardware: the dev machine has no CUDA GPU, so only sample episodes are transcribed locally on CPU. The full 530-hour run still needs a GPU or hosted API (undecided).
- Diarization and host speaker ID: not built yet (pyannote needs a Hugging Face token)

## Decisions

- **LLM backend**: `coach.llm` shells out to the Claude Code CLI (`claude -p`, tool-less, own system prompt, `--json-schema` for structured output), so it runs on the user's Claude subscription with no API key. `FAST_MODEL` (haiku) handles bulk correction; `SMART_MODEL` (sonnet) handles distill and anything needing judgment.
- **Vector store**: LanceDB, embedded under `data/index/`, with its built-in full-text index for the BM25 half. Embeddings are `bge-small-en-v1.5` and reranking is `bge-reranker-base`, both local via fastembed (`coach.embed`).
- **Clustering**: incremental leader clustering at cosine 0.80 (`--threshold`); the LLM review splits off misfits. Lessons with consolidate fields go to `data/concepts/consolidated.json`, never back into `data/lessons/`.
- **Agent**: `coach.serve.agent` runs its own tool loop on `claude -p` structured output (one tool call or the answer per step, max 6 calls). `RIOT_API_KEY` in `.env` enables match history; Data Dragon and Riot responses are cached under `data/cache/`.
- **Eval data**: `src/coach/eval/questions.yaml` and `glossary_refs.yaml`. Items marked `verified: false` were seeded by Claude from transcripts and need a human check; the glossary refs in particular were picked from known corrections, so the clean term error rate is optimistic until real listened-to windows replace them.
- **YouTube bot blocks**: ingest pauses between every attempt and stops after 3 consecutive blocks. Pass `--cookies <file>` with a browser export (cookie files are gitignored).
- **PyAV is pinned below 16**: newer releases dropped an `av.open` argument that faster-whisper 1.2.1 still passes.
