import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

from env.email_env import EmailEnv
from env.sentiment_env import SentimentEnv
from env.priority_env import PriorityEnv

app = Flask(__name__)

# Initialize all envs
envs = [EmailEnv(), SentimentEnv(), PriorityEnv()]
current_env_index = 0


@app.route("/", methods=["GET"])
def home():
    return "OK"


@app.route("/reset", methods=["POST"])
def reset():
    global current_env_index

    try:
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/step", methods=["POST"])
def step():
    global current_env_index

    try:
        data = request.get_json(silent=True) or {}
        action = data.get("action", "support")

        env = envs[current_env_index]
        result = env.step(action)

        obs = result["observation"]
        done = result["done"]

        # If current env finished → move to next env
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

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/state", methods=["GET"])
def state():
    try:
        return jsonify({
            "current_env_index": current_env_index
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    print("🔥 MULTI-TASK ROTATING ENV SERVER", flush=True)
    app.run(host="0.0.0.0", port=7860, debug=False)


if __name__ == "__main__":
    main()
