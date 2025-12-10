import json
from datetime import datetime

path = "results-db/memory_bank.jsonl"

memories = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        memories.append(json.loads(line))

print(f"Loaded {len(memories)} failure memories\n")

for i, m in enumerate(memories, 1):
    ts = datetime.fromtimestamp(m["created_at"])
    print(f"=== Memory {i} ===")
    print(f" Episode ID   : {m['episode_id']}")
    print(f" Scenario     : {m['scenario_type']}")
    print(f" Difficulty   : {m['difficulty']}")
    print(f" Steps        : {m['steps']}")
    print(f" Total reward : {m['total_reward']:.3f}")
    print(f" Time         : {ts}")
    print(" Failure reason:")
    print("  ", m["failure_reason"])
    print(" Lesson:")
    print("  ", m["lesson"])
    print()
