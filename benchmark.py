# benchmark.py

from __future__ import annotations
import os
import time
import csv
from dataclasses import dataclass, asdict
from typing import Literal, List, Dict, Any, Callable

import yaml
import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from scenario.longtail_scenarios import ScenarioConfig, make_longtail_env, LONGTAIL_TYPES
from LLMDriver.memory_bank import MemoryBank
from LLMDriver.reason_reflect import (
    ReflectiveLLMDriver,
    EpisodeLogStep,
    EpisodeResult,
)
from LLMDriver.safety_wrapper import SafetyWrapper, SafetyConfig


# ---------------------------------------------------------------------
# LLM initialisation (mirrors HELLM.py)
# ---------------------------------------------------------------------
OPENAI_CONFIG = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)

if OPENAI_CONFIG["OPENAI_API_TYPE"] == "azure":
    os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG["OPENAI_API_TYPE"]
    os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG["AZURE_API_VERSION"]
    os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG["AZURE_API_BASE"]
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG["AZURE_API_KEY"]
    llm = AzureChatOpenAI(
        deployment_name=OPENAI_CONFIG["AZURE_MODEL"],
        temperature=0,
        max_tokens=1024,
        request_timeout=60,
    )
elif OPENAI_CONFIG["OPENAI_API_TYPE"] == "openai":
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG["OPENAI_KEY"]
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-1106",
        max_tokens=1024,
        request_timeout=60,
    )
else:
    raise ValueError("Unsupported OPENAI_API_TYPE in config.yaml")


DriverKind = Literal[
    "llm_naive",
    "llm_memory",
    "llm_memory_safety",
    "rl_baseline",
]


@dataclass
class EpisodeMetrics:
    driver_kind: str
    scenario_type: str
    difficulty: int
    seed: int
    total_reward: float
    steps: int
    crashed: bool
    truncated: bool


def run_llm_episode(
    env,
    driver: ReflectiveLLMDriver,
    safety: SafetyWrapper | None,
    scenario_type: str,
    difficulty: int,
    seed: int,
    obs_to_text: Callable[[np.ndarray, Dict[str, Any]], str],
) -> EpisodeResult:
    """Run a single episode with an LLM-based driver."""
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0.0
    t = 0
    log_steps: List[EpisodeLogStep] = []

    while not (done or truncated):
        obs_text = obs_to_text(obs, info)
        action_str, llm_raw = driver.decide(obs_text, scenario_type, difficulty)

        if safety is not None:
            safe_action_str = safety.filter_action(action_str, obs)
        else:
            safe_action_str = action_str

        # Map string to discrete action index
        act_idx = env.unwrapped.action_type.actions_indexes.get(safe_action_str, 0)

        obs, reward, done, truncated, info = env.step(act_idx)
        total_reward += reward

        log_steps.append(
            EpisodeLogStep(
                t=t,
                obs_text=obs_text,
                action_str=safe_action_str,
                reward=reward,
                info=info,
                llm_raw=llm_raw,
            )
        )
        t += 1

    episode = EpisodeResult(
        episode_id=f"{scenario_type}-{difficulty}-seed{seed}",
        scenario_type=scenario_type,
        difficulty=difficulty,
        total_reward=total_reward,
        crashed=info.get("crashed", False),
        truncated=truncated,
        steps=t,
        log=log_steps,
    )
    return episode


def run_benchmark(
    driver_kind: DriverKind,
    n_episodes_per_scenario: int = 5,
    difficulty: int = 1,
    scenario_number: int | None = None,
    out_csv: str | None = None,
):
    """Benchmark different driver types.

    Parameters
    ----------
    driver_kind:
        - "llm_naive"          -> LLM without reflection/memory, no safety wrapper
        - "llm_memory"         -> LLM + reflection/memory (no safety)
        - "llm_memory_safety"  -> LLM + reflection/memory + SafetyWrapper
        - "rl_baseline"        -> RL baseline (requires baselines/rl_baseline.py and stable-baselines3)
    n_episodes_per_scenario:
        Number of random seeds (episodes) per scenario.
    difficulty:
        Difficulty level passed to the scenario config.
    scenario_number:
        If None or 0  -> run ALL scenarios in LONGTAIL_TYPES.
        If 1..len(LONGTAIL_TYPES) -> run only that scenario index.
    out_csv:
        Optional explicit path for the CSV file. If None, a timestamped
        filename is created automatically so runs do not overwrite each other.
    """
    os.makedirs("results-benchmark", exist_ok=True)

    # Figure out which scenarios to run
    if scenario_number is None or scenario_number == 0:
        selected_scenarios = list(LONGTAIL_TYPES)
        scenario_tag = "all"
    else:
        if not (1 <= scenario_number <= len(LONGTAIL_TYPES)):
            raise ValueError(
                f"scenario_number must be between 1 and {len(LONGTAIL_TYPES)} "
                f"(got {scenario_number})."
            )
        selected_scenarios = [LONGTAIL_TYPES[scenario_number - 1]]
        scenario_tag = f"s{scenario_number}_{selected_scenarios[0]}"

    # Unique ID for this benchmark run (used if out_csv is not given)
    if out_csv is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        out_csv = (
            f"results-benchmark/benchmark_{driver_kind}_d{difficulty}_{scenario_tag}_{run_id}.csv"
        )

    print("Available scenarios:")
    for idx, name in enumerate(LONGTAIL_TYPES, 1):
        print(f"  {idx}: {name}")
    if scenario_number is None or scenario_number == 0:
        print("Running ALL scenarios. Output ->", out_csv)
    else:
        print(
            f"Running scenario {scenario_number}: {selected_scenarios[0]}  |  Output -> {out_csv}"
        )

    mem_bank = MemoryBank("results-db/memory_bank.jsonl")

    # -----------------------------------------------------------------
    # LLM call wrapper (reuses global `llm`)
    # -----------------------------------------------------------------
    def openai_llm_call(messages: List[Dict[str, str]]) -> str:
        """messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns the assistant's content as a plain string.
        """
        chunks = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            chunks.append(f"[{role.upper()}]\n{content}")
        prompt = "\n\n".join(chunks)
        return llm.predict(prompt)

    driver = ReflectiveLLMDriver(
        llm_call=openai_llm_call,
        memory_bank=mem_bank,
        system_prompt="You are a cautious tactical driver on highway-env.",
        use_reflection=driver_kind in ("llm_memory", "llm_memory_safety"),
    )

    safety = None
    if driver_kind == "llm_memory_safety":
        safety = SafetyWrapper(SafetyConfig())

    # -----------------------------------------------------------------
    # RL policy (for rl_baseline) - OPTIONAL
    # -----------------------------------------------------------------
    rl_model = None
    make_rl_env = None
    if driver_kind == "rl_baseline":
        # Import here (not at top) so script still works without SB3
        from stable_baselines3 import DQN
        from baselines.rl_baseline import make_env as _make_rl_env

        make_rl_env = _make_rl_env
        rl_model = DQN.load("results-rl/dqn_baseline_default")

    # Simple text encoding of observation (you can refine this)
    def obs_to_text(obs, info):
        ego = obs[0]
        vx = ego[3]
        return f"Ego speed ~{vx*3.6:.1f} km/h, {info.get('n_vehicles', 'N/A')} vehicles around."

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(EpisodeMetrics.__annotations__.keys()),
        )
        writer.writeheader()

        for scenario_type in selected_scenarios:
            for ep_idx in range(n_episodes_per_scenario):
                seed = ep_idx

                # ---------------- RL baseline branch ----------------
                if driver_kind == "rl_baseline":
                    env = make_rl_env()
                    obs, info = env.reset(seed=seed)
                    done = False
                    truncated = False
                    total_reward = 0.0
                    t = 0
                    while not (done or truncated):
                        action, _ = rl_model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = env.step(action)
                        total_reward += reward
                        t += 1
                    crashed = info.get("crashed", False)
                    env.close()
                    metrics = EpisodeMetrics(
                        driver_kind=driver_kind,
                        scenario_type=scenario_type,
                        difficulty=difficulty,
                        seed=seed,
                        total_reward=total_reward,
                        steps=t,
                        crashed=crashed,
                        truncated=truncated,
                    )
                    writer.writerow(asdict(metrics))

                # ---------------- LLM driver branch -----------------
                else:
                    cfg = ScenarioConfig(
                        name=f"{scenario_type}_bench",
                        difficulty=difficulty,
                        seed=seed,
                    )
                    env = make_longtail_env(scenario_type, cfg)
                    episode = run_llm_episode(
                        env=env,
                        driver=driver,
                        safety=safety,
                        scenario_type=scenario_type,
                        difficulty=difficulty,
                        seed=seed,
                        obs_to_text=obs_to_text,
                    )
                    if episode.crashed:
                        driver.reflect_and_store(
                            episode,
                            scenario_type=scenario_type,
                            difficulty=difficulty,
                            crash_reason="collision or offroad",
                        )
                    env.close()
                    metrics = EpisodeMetrics(
                        driver_kind=driver_kind,
                        scenario_type=scenario_type,
                        difficulty=difficulty,
                        seed=seed,
                        total_reward=episode.total_reward,
                        steps=episode.steps,
                        crashed=episode.crashed,
                        truncated=episode.truncated,
                    )
                    writer.writerow(asdict(metrics))

# Available scenarios:
#   0: run all scenarios 
#   1: high_speed_cutin
#   2: partial_occlusion
#   3: erratic_braking
#   4: multi_vehicle_merge
#   5: truck_debris
#   6: emergency_merge
#   7: stalled_on_shoulder
#   8: cones_high_density

if __name__ == "__main__":
    # Choose which scenario to run by number:
    #   0 or None -> run ALL scenarios
    #   1..len(LONGTAIL_TYPES) -> run only that scenario index
    SCENARIO_NUMBER = 0  # <-- change this to e.g. 3 to run only scenario #3

    # Default: benchmark LLM + Memory + Safety
    run_benchmark(
        driver_kind="llm_memory_safety",
        n_episodes_per_scenario=3,
        difficulty=1,
        scenario_number=SCENARIO_NUMBER,
    )
