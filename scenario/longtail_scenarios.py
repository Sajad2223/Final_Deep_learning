# scenario/longtail_scenarios.py

from dataclasses import dataclass
from typing import Dict, Any, Optional
import gymnasium as gym
import highway_env  # noqa: F401  # needed to register envs

LONGTAIL_TYPES = [
    "high_speed_cutin",
    "partial_occlusion",
    "erratic_braking",
    "multi_vehicle_merge",
    "truck_debris",
    "emergency_merge",
    "stalled_on_shoulder",
    "cones_high_density",
]


@dataclass
class ScenarioConfig:
    name: str
    base_env_id: str = "highway-v0"
    # difficulty = 0 (normal), 1 (hard), 2 (extreme)
    difficulty: int = 0
    seed: Optional[int] = None
    # arbitrary tag so you can log easily
    tag: Optional[str] = None


def _apply_base_difficulty(env, cfg: ScenarioConfig):
    """
    Tweak generic difficulty knobs: vehicles_count, vehicles_density, lanes_count, duration, etc.
    """
    c = env.config

    if cfg.difficulty == 0:  # normal
        c["lanes_count"] = 4
        c["vehicles_density"] = 1.5
        c["vehicles_count"] = 40
    elif cfg.difficulty == 1:  # hard
        c["lanes_count"] = 5
        c["vehicles_density"] = 2.0
        c["vehicles_count"] = 60
    else:  # extreme
        c["lanes_count"] = 6
        c["vehicles_density"] = 2.5
        c["vehicles_count"] = 80

    c["duration"] = 40
    c["collision_reward"] = -1.0
    c["reward_speed_range"] = [20, 30]
    c["policy_frequency"] = 1
    c["simulation_frequency"] = 15


def _configure_high_speed_cutin(env, cfg: ScenarioConfig):
    """
    Dense traffic + aggressive merging behind/next to ego.
    You mostly control how spawn speeds and gaps are sampled through env config.
    """
    _apply_base_difficulty(env, cfg)
    c = env.config
    c["vehicles_density"] += 0.5
    # Make others use IDM behavior (default) but you can switch to more aggressive vehicle model if needed
    c["other_vehicles_type"] = "highway_env.vehicle.behavior.IDMVehicle"
    # Encourage higher speed so cut-ins are sharper
    c["reward_speed_range"] = [25, 35]


def _configure_partial_occlusion(env, cfg: ScenarioConfig):
    """
    Emulate occlusion by lower vehicle_count but high speed and lane changes.
    Vision occlusion in highway-env is tricky; we approximate with lateral clutter & close vehicles.
    """
    _apply_base_difficulty(env, cfg)
    c = env.config
    c["vehicles_density"] += 0.2
    c["show_trajectories"] = False   # no visual help
    c["offscreen_rendering"] = False


def _configure_erratic_braking(env, cfg: ScenarioConfig):
    _apply_base_difficulty(env, cfg)
    c = env.config
    # You can later define a custom vehicle class that randomly brakes.
    # For now: more dense + lower reward for high speed -> encourages more cautious behavior.
    c["vehicles_density"] += 0.7
    c["reward_speed_range"] = [15, 25]


def _configure_multi_vehicle_merge(env, cfg: ScenarioConfig):
    """
    Use merge-v0 environment; ego on main road, many vehicles merging.
    """
    cfg.base_env_id = "merge-v0"
    _apply_base_difficulty(env, cfg)
    c = env.config
    c["vehicles_density"] += 0.5


def _configure_truck_debris(env, cfg: ScenarioConfig):
    _apply_base_difficulty(env, cfg)
    # For real "debris" you’d need to hack highway-env; approximated here by a very slow stopped car
    # spawned ahead in ego’s lane (you can later subclass env._make_road()).
    c = env.config
    c["collision_reward"] = -2.0


def _configure_emergency_merge(env, cfg: ScenarioConfig):
    _apply_base_difficulty(env, cfg)
    # Later you can mark one vehicle as emergency and design ad-hoc rule in LLM prompt
    c = env.config


def _configure_stalled_on_shoulder(env, cfg: ScenarioConfig):
    _apply_base_difficulty(env, cfg)
    c = env.config
    # Use more lanes; stalled car on far right lane approximates shoulder hazard for lane changes
    c["lanes_count"] = max(c.get("lanes_count", 4), 4)


def _configure_cones_high_density(env, cfg: ScenarioConfig):
    _apply_base_difficulty(env, cfg)
    c = env.config
    # Approximated as extremely high density and reward shaping that penalizes lane changes
    c["vehicles_density"] += 1.0
    c["lane_change_reward"] = -0.1


CONFIGURATORS = {
    "high_speed_cutin": _configure_high_speed_cutin,
    "partial_occlusion": _configure_partial_occlusion,
    "erratic_braking": _configure_erratic_braking,
    "multi_vehicle_merge": _configure_multi_vehicle_merge,
    "truck_debris": _configure_truck_debris,
    "emergency_merge": _configure_emergency_merge,
    "stalled_on_shoulder": _configure_stalled_on_shoulder,
    "cones_high_density": _configure_cones_high_density,
}


def make_longtail_env(scenario_type: str, cfg: ScenarioConfig):
    assert scenario_type in CONFIGURATORS, f"Unknown scenario_type: {scenario_type}"
    env = gym.make(cfg.base_env_id)
    CONFIGURATORS[scenario_type](env, cfg)
    if cfg.seed is not None:
        env.reset(seed=cfg.seed)
    else:
        env.reset()
    return env


def list_longtail_scenarios() -> Dict[str, Any]:
    """Small helper if you want to show them in CLI."""
    return {
        name: {"description": name.replace("_", " ").title()}
        for name in LONGTAIL_TYPES
    }
