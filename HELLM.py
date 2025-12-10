import os
import yaml
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from scenario.scenario import Scenario
from LLMDriver.driverAgent import DriverAgent
from LLMDriver.outputAgent import OutputParser
from LLMDriver.customTools import (
    getAvailableActions,
    getAvailableLanes,
    getLaneInvolvedCar,
    isChangeLaneConflictWithCar,
    isAccelerationConflictWithCar,
    isKeepSpeedConflictWithCar,
    isDecelerationSafe,
    isActionSafe,
)
from LLMDriver.reason_reflect import ReflectiveLLMDriver, EpisodeLogStep, EpisodeResult
from LLMDriver.memory_bank import MemoryBank

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
        model_name="gpt-3.5-turbo-1106",  # or any other model with 8k+ context
        max_tokens=1024,
        request_timeout=60,
    )


# simple wrapper so ReflectiveLLMDriver can call the same llm
def llm_call_for_reflection(messages):
    """
    Convert OpenAI-style message list into a single prompt string for LangChain llm.

    messages: list of dicts like {"role": "system"|"user"|"assistant", "content": str}
    returns: assistant content string
    """
    chunks = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        chunks.append(f"[{role.upper()}]\n{content}")
    prompt = "\n\n".join(chunks)
    # ChatOpenAI / AzureChatOpenAI both support .predict on a single string
    response = llm.predict(prompt)
    return response


# base setting
vehicleCount = 15

# environment setting
config = {
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "normalize": False,
        "vehicles_count": vehicleCount,
        "see_behind": True,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": np.linspace(0, 32, 9),
    },
    "duration": 40,
    "vehicles_density": 2,
    "show_trajectories": True,
    "render_agent": True,
}

env = gym.make("highway-v0", render_mode="rgb_array")
env.configure(config)
env = RecordVideo(
    env,
    "./results-video",
    name_prefix=f"highwayv0",
)
env.unwrapped.set_record_video_wrapper(env)
obs, info = env.reset()
env.render()

# scenario and driver agent setting
if not os.path.exists("results-db/"):
    os.mkdir("results-db")
database = "results-db/highwayv0.db"
sce = Scenario(vehicleCount, database)
toolModels = [
    getAvailableActions(env),
    getAvailableLanes(sce),
    getLaneInvolvedCar(sce),
    isChangeLaneConflictWithCar(sce),
    isAccelerationConflictWithCar(sce),
    isKeepSpeedConflictWithCar(sce),
    isDecelerationSafe(sce),
    isActionSafe(),
]
DA = DriverAgent(llm, toolModels, sce, verbose=True)
outputParser = OutputParser(sce, llm)

# memory bank + reflective driver (for failure analysis)
memory_bank = MemoryBank("results-db/memory_bank.jsonl")
reflective_driver = ReflectiveLLMDriver(
    llm_call=llm_call_for_reflection,
    memory_bank=memory_bank,
    system_prompt="You analyse highway driving behavior and extract lessons from failures.",
    max_memory_k=5,
    use_reflection=True,
)

output = None
done = False
truncated = False
frame = 0
total_reward = 0.0
episode_log = []
last_info = {}
scenario_type = "highwayv0"
difficulty = 1  # you can change / parametrize this if you like

try:
    while not (done or truncated):
        # update scenario with latest observation
        sce.upateVehicles(obs, frame)

        # run HELLM agent and parse its output
        DA.agentRun(output)
        da_output = DA.exportThoughts()
        output = outputParser.agentRun(da_output)

        # basic textual summary of state for the memory / reflection system
        obs_text = (
            f"frame={frame}, obs={np.array2string(obs, precision=2, separator=',')}\n"
            f"da_output={str(da_output)[:400]}"
        )

        env.render()
        env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame()

        # gymnasium step: obs, reward, terminated, truncated, info
        obs, reward, done, truncated, info = env.step(output["action_id"])
        last_info = info
        total_reward += reward

        # log this step for potential failure reflection later
        episode_log.append(
            EpisodeLogStep(
                t=frame,
                obs_text=obs_text,
                action_str=str(output.get("action_id", "NA")),
                reward=reward,
                info=info,
                llm_raw=str(da_output),
            )
        )

        print(output)
        frame += 1
finally:
    # build episode summary and, if needed, store a failure memory
    crashed = bool(last_info.get("crashed", False))
    episode = EpisodeResult(
        episode_id=f"{scenario_type}-single-run",
        scenario_type=scenario_type,
        difficulty=difficulty,
        total_reward=total_reward,
        crashed=crashed,
        truncated=truncated,
        steps=frame,
        log=episode_log,
    )

    if crashed:
        crash_reason = last_info.get("crash_reason", "collision or off-road event")
        reflective_driver.reflect_and_store(
            episode=episode,
            scenario_type=scenario_type,
            difficulty=difficulty,
            crash_reason=crash_reason,
        )

    env.close()
