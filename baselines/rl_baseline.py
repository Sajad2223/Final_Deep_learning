# baselines/rl_baseline.py

import os
from typing import Optional
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure


def make_env(env_id: str = "highway-v0", scenario_type: Optional[str] = None, difficulty: int = 0):
    env = gym.make(env_id)
    c = env.config

    # Basic difficulty knobs like before
    if difficulty == 0:
        c["lanes_count"] = 4
        c["vehicles_density"] = 1.5
        c["vehicles_count"] = 40
    elif difficulty == 1:
        c["lanes_count"] = 5
        c["vehicles_density"] = 2.0
        c["vehicles_count"] = 60
    else:
        c["lanes_count"] = 6
        c["vehicles_density"] = 2.5
        c["vehicles_count"] = 80

    c["duration"] = 40
    c["collision_reward"] = -1.0

    # You can specialise per scenario_type if you want, similar to longtail_scenarios.py

    env.reset()
    return env


def train_dqn_baseline(
    total_timesteps: int = 200_000,
    scenario_type: Optional[str] = None,
    difficulty: int = 0,
    save_dir: str = "results-rl",
):
    os.makedirs(save_dir, exist_ok=True)

    env = make_env("highway-v0", scenario_type, difficulty)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tb"),
    )

    new_logger = configure(save_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(save_dir, f"dqn_baseline_{scenario_type or 'default'}"))

    env.close()


def evaluate_policy(model_path: str, n_episodes: int = 20):
    model = DQN.load(model_path)
    env = make_env()
    rewards = []
    crashes = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_r = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_r += reward
        rewards.append(ep_r)
        crashes += int(info.get("crashed", False))

    env.close()
    return {
        "mean_reward": sum(rewards) / len(rewards),
        "crash_rate": crashes / n_episodes,
    }


if __name__ == "__main__":
    train_dqn_baseline()
