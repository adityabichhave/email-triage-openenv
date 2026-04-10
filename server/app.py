import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

from env.email_env import EmailEnv
from env.sentiment_env import SentimentEnv
from env.priority_env import PriorityEnv

app = Flask(__name__)

# Create env instances
email_env = EmailEnv()
sentiment_env = SentimentEnv()
priority_env = PriorityEnv()


@app.route("/", methods=["GET"])
def home():
    return "OK"


# =========================
# TASK 1: EMAIL
# =========================
@app.route("/email/reset", methods=["POST"])
def email_reset():
    res = email_env.reset()
    obs = res["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(res["reward"].value)},
        "done": bool(res["done"]),
        "info": {"score": float(res["reward"].value)}
    })


@app.route("/email/step", methods=["POST"])
def email_step():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "support")

    result = email_env.step(action)
    obs = result["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(result["reward"].value)},
        "done": bool(result["done"]),
        "info": {"score": float(result["reward"].value)}
    })


# =========================
# TASK 2: SENTIMENT
# =========================
@app.route("/sentiment/reset", methods=["POST"])
def sentiment_reset():
    res = sentiment_env.reset()
    obs = res["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(res["reward"].value)},
        "done": bool(res["done"]),
        "info": {"score": float(res["reward"].value)}
    })


@app.route("/sentiment/step", methods=["POST"])
def sentiment_step():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "positive")

    result = sentiment_env.step(action)
    obs = result["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(result["reward"].value)},
        "done": bool(result["done"]),
        "info": {"score": float(result["reward"].value)}
    })


# =========================
# TASK 3: PRIORITY
# =========================
@app.route("/priority/reset", methods=["POST"])
def priority_reset():
    res = priority_env.reset()
    obs = res["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(res["reward"].value)},
        "done": bool(res["done"]),
        "info": {"score": float(res["reward"].value)}
    })


@app.route("/priority/step", methods=["POST"])
def priority_step():
    data = request.get_json(silent=True) or {}
    action = data.get("action", "low")

    result = priority_env.step(action)
    obs = result["observation"]

    return jsonify({
        "observation": {"email": obs.email},
        "reward": {"value": float(result["reward"].value)},
        "done": bool(result["done"]),
        "info": {"score": float(result["reward"].value)}
    })


# =========================
# STATE (OPTIONAL)
# =========================
@app.route("/state", methods=["GET"])
def state():
    return jsonify({
        "email": email_env.state(),
        "sentiment": sentiment_env.state(),
        "priority": priority_env.state()
    })


def main():
    print("🔥 3-TASK SERVER RUNNING", flush=True)
    app.run(host="0.0.0.0", port=7860, debug=False)


if __name__ == "__main__":
    main()
