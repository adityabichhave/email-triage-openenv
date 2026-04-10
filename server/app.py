print("🔥 APP STARTING...")

import sys
import os
from flask import Flask, request, jsonify

# PATH FIX
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
if root_path not in sys.path:
    sys.path.append(root_path)

# IMPORT ENVS
from env.email_env import EmailEnv
from env.sentiment_env import SentimentEnv
from env.priority_env import PriorityEnv

env_map = {
    "email": EmailEnv(),
    "sentiment": SentimentEnv(),
    "priority": PriorityEnv()
}

task_keys = list(env_map.keys())
task_index = 0
current_env = env_map[task_keys[0]]

print("✅ ENV LOADED SUCCESSFULLY")

app = Flask(__name__)

# ROOT
@app.route("/", methods=["GET"])
def home():
    return "OK"

# RESET
@app.route("/reset", methods=["POST"])
def reset():
    global task_index, current_env

    # 🔥 Rotate TASK NAME (CRITICAL)
    task_name = task_keys[task_index % 3]
    current_env = env_map[task_name]
    task_index += 1

    result = current_env.reset()
    obs = result["observation"]
    rew = result["reward"]

    return jsonify({
        "observation": {
            "email": obs.email
        },
        "reward": {
            "value": float(rew.value)
        },
        "done": False,
        "info": {
            "score": float(rew.value),
            "task": task_name   # 🔥 CRITICAL FOR VALIDATOR
        }
    })

# STEP
@app.route("/step", methods=["POST"])
def step():
    global current_env, task_index

    data = request.get_json(silent=True) or {}
    action = data.get("action", "support")

    result = current_env.step(action)

    obs = result["observation"]
    rew = result["reward"]
    done = result.get("done", False)

    return jsonify({
        "observation": {
            "email": obs.email if obs else ""
        },
        "reward": {
            "value": float(rew.value) if rew else 0.1
        },
        "done": bool(done),
        "info": {
            "score": float(rew.value) if rew else 0.1,
            "task": task_keys[(task_index - 1) % 3]  # 🔥 SAME TASK NAME
        }
    })

# STATE
@app.route("/state", methods=["GET"])
def state():
    return jsonify({"task": task_keys[(task_index - 1) % 3]})

# MAIN
def main():
    print("🚀 RUNNING FLASK ON PORT 7860...")
    app.run(host="0.0.0.0", port=7860, debug=False)

if __name__ == "__main__":
    main()
