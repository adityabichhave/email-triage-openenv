print("🔥 APP STARTING...")

import sys
import os

# ✅ FIX PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

# -------- SAFE IMPORT --------
try:
    from env import EmailEnv
    env = EmailEnv()
    print("✅ ENV LOADED")
except Exception as e:
    print("❌ ERROR LOADING ENV:", e)
    env = None

app = Flask(__name__)


# -------- RESET --------
@app.route("/reset", methods=["POST"])
def reset():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    result = env.reset()

    return jsonify({
        "observation": {
            "email": result["observation"].email
        },
        "reward": {
            "value": 0.0
        },
        "done": False,
        "info": {}
    })


# -------- STEP --------
@app.route("/step", methods=["POST"])
def step():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    data = request.get_json(force=True)
    action = data.get("action")

    result = env.step(action)

    obs = result["observation"]
    rew = result["reward"]

    return jsonify({
        "observation": {
            "email": obs.email if obs else ""
        },
        "reward": {
            "value": rew.value if rew else 0.0
        },
        "done": result.get("done", False),
        "info": result.get("info", {})
    })


# -------- STATE --------
@app.route("/state", methods=["GET"])
def state():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    return jsonify(env.state())


# -------- ROOT --------
@app.route("/", methods=["GET"])
def home():
    return "OK"


# -------- MAIN --------
def main():
    print("🚀 RUNNING FLASK...")
    app.run(host="0.0.0.0", port=7860)


# -------- ENTRYPOINT --------
if __name__ == "__main__":
    main()
