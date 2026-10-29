"""On-disk records for host voiceprints and per-episode speaker maps."""

from typing import Literal

import yaml
from pydantic import BaseModel

from coach.config import HOSTS_DIR


class Voiceprint(BaseModel):
    """Stored as `hosts/<host>/voiceprint.json`."""

    host: str
    model: str  # embedding model; must match the diarization pipeline's
    source: Literal["enrolled", "bootstrapped"]
    embedding: list[float]  # unit length
    # enrolled: the reference clips; bootstrapped: the episodes it was averaged from
    from_ids: list[str]
    chunks: int  # embeddings averaged


class HostConfig(BaseModel):
    """`hosts/hosts.yaml`."""

    hosts: list[str]
    by_elimination: str | None = None


Method = Literal["voiceprint", "elimination", "ambiguous", "short", "no_match"]


class SpeakerMatch(BaseModel):
    label: str  # diarization label
    seconds: float
    host: str  # host id, "guest_unknown", or "uncertain"
    method: Method
    similarity: dict[str, float]  # cosine to each voiceprint


class SpeakerMap(BaseModel):
    """Stored as `data/speakers/<episode_id>.json`."""

    episode_id: str
    guest_episode: bool
    voiceprints: dict[str, str]  # host -> voiceprint source used
    speakers: list[SpeakerMatch]

    def hosts_by_label(self) -> dict[str, str]:
        return {s.label: s.host for s in self.speakers}


def load_config(hosts_dir=HOSTS_DIR) -> HostConfig:
    return HostConfig(**yaml.safe_load((hosts_dir / "hosts.yaml").read_text(encoding="utf-8")))


def load_voiceprints(config: HostConfig, hosts_dir=HOSTS_DIR) -> dict[str, Voiceprint]:
    out = {}
    for host in config.hosts:
        path = hosts_dir / host / "voiceprint.json"
        if path.exists():
            out[host] = Voiceprint.model_validate_json(path.read_text(encoding="utf-8"))
    return out
