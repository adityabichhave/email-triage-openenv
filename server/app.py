import os
from flask import Flask, request, jsonify

from env.email_env import EmailEnv
from env.sentiment_env import SentimentEnv
from env.priority_env import PriorityEnv

app = Flask(__name__)

# 🔥 REGISTER ALL TASKS HERE
ENV_MAP = {
    "email": EmailEnv,
    "sentiment": SentimentEnv,
    "priority": PriorityEnv
}

env_instances = {}


def get_env(task_name):
    if task_name not in ENV_MAP:
        raise Exception(f"Invalid task: {task_name}")

    if task_name not in env_instances:
        env_instances[task_name] = ENV_MAP[task_name]()

    return env_instances[task_name]


@app.route("/", methods=["GET"])
def home():
    return "OK"


@app.route("/reset", methods=["POST"])
def reset():
    try:
        data = request.get_json(force=True) or {}
        task = data.get("task", "email")

        env = get_env(task)
        res = env.reset()
        obs = res["observation"]

        return jsonify({
            "observation": {"email": obs.email},
            "reward": {"value": float(res["reward"].value)},
            "done": bool(res["done"]),
            "info": {"score": float(res["reward"].value), "task": task}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/step", methods=["POST"])
def step():
    try:
        data = request.get_json(force=True)
        task = data.get("task", "email")
        action = data.get("action", "support")

        env = get_env(task)
        result = env.step(action)
        obs = result["observation"]

        return jsonify({
            "observation": {"email": obs.email},
            "reward": {"value": float(result["reward"].value)},
            "done": bool(result["done"]),
            "info": {
                "score": float(result["reward"].value),
                "task": task
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/state", methods=["GET"])
def state():
    try:
        task = request.args.get("task", "email")
        env = get_env(task)
        return jsonify(env.state())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    print("🔥 MULTI-TASK APP STARTED", flush=True)
    app.run(host="0.0.0.0", port=7860, debug=False)


if __name__ == "__main__":
    main()
