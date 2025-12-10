# LLMDriver/memory_bank.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Iterable, Optional
import json
import os
import time
from difflib import SequenceMatcher


@dataclass
class EpisodeMemory:
    episode_id: str
    scenario_type: str
    difficulty: int
    is_failure: bool
    total_reward: float
    steps: int
    failure_reason: str
    lesson: str
    episode_summary: str
    created_at: float  # unix timestamp


class MemoryBank:
    """
    Very simple JSONL-based memory store.
    For small projects this is enough and easy to inspect.
    """

    def __init__(self, path: str = "results-db/memory_bank.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._memories: List[EpisodeMemory] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._memories.append(EpisodeMemory(**data))

    def _save_append(self, memory: EpisodeMemory) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(memory)) + "\n")

    def add(self, memory: EpisodeMemory) -> None:
        self._memories.append(memory)
        self._save_append(memory)

    @property
    def memories(self) -> List[EpisodeMemory]:
        return list(self._memories)

    def query(
        self,
        query_text: str,
        scenario_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[EpisodeMemory]:
        """
        Super crude lexical similarity using difflib.
        Enough for < 1k memories. Replace with embeddings if you want.
        """
        candidates: Iterable[EpisodeMemory] = self._memories
        if scenario_type is not None:
            candidates = [m for m in candidates if m.scenario_type == scenario_type]

        scored = []
        for m in candidates:
            text = m.episode_summary + "\n" + m.lesson + "\n" + m.failure_reason
            score = SequenceMatcher(
                a=query_text.lower(), b=text.lower()
            ).ratio()
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for score, m in scored[:top_k] if score > 0.2]

    def __len__(self) -> int:
        return len(self._memories)


def make_memory_from_failure(
    episode_id: str,
    scenario_type: str,
    difficulty: int,
    total_reward: float,
    steps: int,
    failure_reason: str,
    lesson: str,
    episode_summary: str,
) -> EpisodeMemory:
    return EpisodeMemory(
        episode_id=episode_id,
        scenario_type=scenario_type,
        difficulty=difficulty,
        is_failure=True,
        total_reward=total_reward,
        steps=steps,
        failure_reason=failure_reason,
        lesson=lesson,
        episode_summary=episode_summary,
        created_at=time.time(),
    )
