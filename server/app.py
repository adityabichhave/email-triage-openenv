import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

from env.email_env import EmailEnv
from env.sentiment_env import SentimentEnv
from env.priority_env import PriorityEnv

app = Flask(__name__)

envs = [EmailEnv(), SentimentEnv(), PriorityEnv()]
current_env_index = 0


@app.route("/", methods=["GET"])
def home():
    return "OK"


@app.route("/reset", methods=["POST"])
def reset():
    global current_env_index
    current_env_index = 0

    env = envs[current_env_index]
    res = env.reset()
    obs = res["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(res["reward"].value)},
        "done": False,
        "info": {"score": float(res["reward"].value)}
    })


@app.route("/step", methods=["POST"])
def step():
    global current_env_index

    data = request.get_json(silent=True) or {}
    action = data.get("action", "support")

    env = envs[current_env_index]
    result = env.step(action)

    obs = result["observation"]
    done = result["done"]

    if done:
        current_env_index += 1
        if current_env_index < len(envs):
            next_env = envs[current_env_index]
            next_res = next_env.reset()
            obs = next_res["observation"]
            done = False

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(result["reward"].value)},
        "done": bool(done),
        "info": {"score": float(result["reward"].value)}
    })


@app.route("/state", methods=["GET"])
def state():
    return jsonify({"current_env_index": current_env_index})


def main():
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
