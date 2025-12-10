# LLMDriver/reason_reflect.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Tuple

import json
import re

from .memory_bank import MemoryBank, make_memory_from_failure


@dataclass
class EpisodeLogStep:
    t: int
    obs_text: str
    action_str: str
    reward: float
    info: Dict[str, Any]
    llm_raw: str


@dataclass
class EpisodeResult:
    episode_id: str
    scenario_type: str
    difficulty: int
    total_reward: float
    crashed: bool
    truncated: bool
    steps: int
    log: List[EpisodeLogStep]


# Signature for the LLM backend: OpenAI-style messages -> assistant string.
LLMFn = Callable[[List[Dict[str, str]]], str]


class ReflectiveLLMDriver:
    """LLM driver with optional failure reflection + memory retrieval."""

    VALID_ACTIONS = ["IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"]

    def __init__(
        self,
        llm_call: LLMFn,
        memory_bank: MemoryBank,
        system_prompt: str,
        max_memory_k: int = 5,
        use_reflection: bool = True,
    ) -> None:
        self.llm_call = llm_call
        self.memory_bank = memory_bank
        self.system_prompt = system_prompt
        self.max_memory_k = max_memory_k
        self.use_reflection = use_reflection

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def _extract_action(self, llm_output: str) -> str:
        """Robustly extract one of the VALID_ACTIONS tokens from model output."""
        upper = llm_output.upper()

        # 1) Exact token search
        m = re.search(r"\b(IDLE|FASTER|SLOWER|LANE_LEFT|LANE_RIGHT)\b", upper)
        if m:
            return m.group(1)

        # 2) Common phrase fallbacks
        if "KEEP LANE" in upper or "KEEP SPEED" in upper:
            return "IDLE"
        if "SLOW" in upper and "FASTER" not in upper:
            return "SLOWER"
        if "ACCEL" in upper or "SPEED UP" in upper:
            return "FASTER"

        # 3) Ultimate fallback: safest choice
        return "IDLE"

    def decide(
        self,
        obs_text: str,
        scenario_type: str,
        difficulty: int,
    ) -> Tuple[str, str]:
        """Return (action_str, raw_llm_output)."""

        # 1) Retrieve similar past failures
        retrieved = self.memory_bank.query(
            query_text=obs_text,
            scenario_type=scenario_type,
            top_k=self.max_memory_k,
        )

        if retrieved:
            memory_lines: List[str] = []
            for m in retrieved:
                memory_lines.append(
                    f"- [{m.scenario_type} / diff={m.difficulty}] "
                    f"Lesson: {m.lesson} (failure: {m.failure_reason})"
                )
            memory_block = (
                "Here are previous failure memories that may help you:\n"
                + "\n".join(memory_lines)
            )
        else:
            memory_block = "No relevant past failures found."

        user_prompt = f"""Current scenario type: {scenario_type}, difficulty: {difficulty}

Environment summary:
{obs_text}

{memory_block}

You are a tactical highway-driving policy.
You must choose a SINGLE high-level action token from the set:
{self.VALID_ACTIONS}

Return ONLY the chosen token, with no explanations, JSON, or extra words.
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_output = self.llm_call(messages).strip()
        action_str = self._extract_action(llm_output)
        return action_str, llm_output

    # ------------------------------------------------------------------
    # Reflection
    # ------------------------------------------------------------------
    def reflect_and_store(
        self,
        episode: EpisodeResult,
        scenario_type: str,
        difficulty: int,
        crash_reason: str,
    ) -> None:
        """Create and store a failure memory if reflection is enabled."""
        if not self.use_reflection or not episode.crashed:
            return

        # Compress last N steps into a short summary
        N = 20
        lines: List[str] = []
        for step in episode.log[-N:]:
            obs_snip = step.obs_text.replace("\n", " ")
            if len(obs_snip) > 120:
                obs_snip = obs_snip[:117] + "..."
            lines.append(
                f"t={step.t}, obs='{obs_snip}', action={step.action_str}, "
                f"reward={step.reward:.3f}"
            )
        episode_summary = "\n".join(lines)

        reflection_prompt = f"""We executed a highway-driving episode in scenario_type={scenario_type},
difficulty={difficulty}, and the episode ended with a crash.

Crash reason from the environment or evaluator:
{crash_reason}

Here is a brief log of the last steps before the crash:
{episode_summary}

You are an expert driving instructor for an autonomous agent.
In 1–2 sentences, explain the root cause of this failure.
Then, in another 1–2 sentences, state a concise LESSON the agent should
remember to avoid this in the future.

Respond in the following JSON format:
{{"root_cause": "...", "lesson": "..."}}
"""

        messages = [
            {"role": "system", "content": "You analyse failures and extract lessons."},
            {"role": "user", "content": reflection_prompt},
        ]

        raw = self.llm_call(messages)

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object in reflection output.")
            data = json.loads(match.group(0))
            root_cause = data.get("root_cause", crash_reason)
            lesson = data.get("lesson", "Be more conservative.")
        except Exception:
            root_cause = crash_reason
            lesson = (
                "Be more conservative around hazards: slow earlier and increase headway."
            )

        mem = make_memory_from_failure(
            episode_id=episode.episode_id,
            scenario_type=scenario_type,
            difficulty=difficulty,
            total_reward=episode.total_reward,
            steps=episode.steps,
            failure_reason=root_cause,
            lesson=lesson,
            episode_summary=episode_summary,
        )
        self.memory_bank.add(mem)
