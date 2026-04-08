print("🔥 APP STARTING...")

import sys
import os

# ✅ FIX PATH (VERY IMPORTANT)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

# -------- SAFE IMPORT --------
try:
    from server.env import EmailEnv
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

    return jsonify({
        "observation": {
            "email": result["observation"].email
        },
        "reward": {
            "value": result["reward"].value
        },
        "done": result["done"],
        "info": result["info"]
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


# ✅ REQUIRED MAIN FUNCTION
def main():
    print("🚀 RUNNING FLASK...")
    app.run(host="0.0.0.0", port=7860)


# ✅ ENTRYPOINT
if __name__ == "__main__":
    main()
