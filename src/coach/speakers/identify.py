"""Match each episode's diarized speakers to host voiceprints.

Thresholds are starting guesses for pyannote's wespeaker embeddings. Calibrate them on
hand-checked episodes before trusting attribution.
"""

import re

import numpy as np

from coach.speakers.models import HostConfig, SpeakerMap, SpeakerMatch, Voiceprint
from coach.transcribe.diarize import EMBEDDING_MODEL
from coach.transcribe.models import DiarizedSpeaker

THRESHOLD = 0.5  # cosine; a speaker below this for every host matches none of them
MARGIN = 0.1  # the best host must beat the runner-up by this much
MIN_SECONDS = 120  # less talk time than this is too little to identify reliably
MIN_BOOTSTRAP = 5  # elimination episodes needed before averaging a voiceprint from them
GUEST_TITLE = re.compile(r"\b(ft|feat)\b\.?|\bfeaturing\b|\bguest", re.IGNORECASE)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def is_guest_episode(title: str, speakers: list[DiarizedSpeaker], config: HostConfig) -> bool:
    main = sum(s.seconds >= MIN_SECONDS for s in speakers)
    return bool(GUEST_TITLE.search(title)) or main > len(config.hosts)


def match(speaker: DiarizedSpeaker, voiceprints: dict[str, Voiceprint]) -> SpeakerMatch:
    sims = {}
    if speaker.embedding is not None:
        e = unit(speaker.embedding)
        sims = {h: round(float(e @ unit(vp.embedding)), 4) for h, vp in voiceprints.items()}
    out = SpeakerMatch(
        label=speaker.label,
        seconds=speaker.seconds,
        host="uncertain",
        method="short",
        similarity=sims,
    )
    if speaker.seconds < MIN_SECONDS or speaker.embedding is None:
        return out
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] < THRESHOLD:
        out.host, out.method = "guest_unknown", "no_match"
    elif len(ranked) > 1 and ranked[0][1] - ranked[1][1] < MARGIN:
        out.method = "ambiguous"  # stays uncertain: a wrong host is worse than none
    else:
        out.host, out.method = ranked[0][0], "voiceprint"
    return out


def identify(
    episode_id: str,
    title: str,
    speakers: list[DiarizedSpeaker],
    voiceprints: dict[str, Voiceprint],
    config: HostConfig,
) -> SpeakerMap:
    guest = is_guest_episode(title, speakers, config)
    matches = [match(s, voiceprints) for s in speakers]
    elim = config.by_elimination
    if elim and elim not in voiceprints and not guest:
        # Only a speaker who is clearly none of the known hosts; one who is merely below the
        # threshold may be a known host that diarization split in two.
        leftover = [
            m
            for m in matches
            if m.method == "no_match" and max(m.similarity.values(), default=0) < THRESHOLD - MARGIN
        ]
        if len(leftover) == 1 and len(leftover) == sum(m.method == "no_match" for m in matches):
            leftover[0].host, leftover[0].method = elim, "elimination"
    return SpeakerMap(
        episode_id=episode_id,
        guest_episode=guest,
        voiceprints={h: vp.source for h, vp in voiceprints.items()},
        speakers=matches,
    )


def bootstrap(
    host: str,
    maps: list[SpeakerMap],
    speakers: dict[str, list[DiarizedSpeaker]],
    config: HostConfig,
) -> Voiceprint | None:
    """Average a voiceprint for `host` from episodes where they were labeled by elimination
    and every other host was matched by voiceprint, so the leftover really is them."""
    others = set(config.hosts) - {host}
    embeddings, episodes = [], []
    for m in maps:
        found = {s.host for s in m.speakers if s.method == "voiceprint"}
        elim = [s for s in m.speakers if s.method == "elimination"]
        if not elim or not others <= found:
            continue
        by_label = {s.label: s for s in speakers[m.episode_id]}
        embeddings.append(unit(by_label[elim[0].label].embedding))
        episodes.append(m.episode_id)
    if len(embeddings) < MIN_BOOTSTRAP:
        return None
    stack = np.stack(embeddings)
    mean = unit(stack.mean(axis=0))
    keep = stack @ mean >= THRESHOLD  # drop outliers: a guest the title didn't mention
    if keep.sum() < MIN_BOOTSTRAP:
        return None
    return Voiceprint(
        host=host,
        model=EMBEDDING_MODEL,
        source="bootstrapped",
        embedding=[float(x) for x in unit(stack[keep].mean(axis=0))],
        from_ids=[e for e, k in zip(episodes, keep, strict=True) if k],
        chunks=int(keep.sum()),
    )
