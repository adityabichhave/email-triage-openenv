print("🔥 APP STARTING...")

import sys
import os
from flask import Flask, request, jsonify

# ✅ PATH FIX
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)
if root_path not in sys.path:
    sys.path.append(root_path)

# -------- IMPORT ENVS --------
try:
    from env.email_env import EmailEnv
    from env.sentiment_env import SentimentEnv
    from env.priority_env import PriorityEnv

    envs = [EmailEnv(), SentimentEnv(), PriorityEnv()]
    current_env_index = 0
    env = envs[current_env_index]

    print("✅ ENV LOADED SUCCESSFULLY")

except Exception as e:
    print("❌ ERROR LOADING ENV:", e)
    import traceback
    traceback.print_exc()
    envs = []
    env = None

app = Flask(__name__)

# -------- ROOT --------
@app.route("/", methods=["GET"])
def home():
    return "OK"

# -------- RESET --------
@app.route("/reset", methods=["POST"])
def reset():
    global current_env_index, env

    if not envs:
        return jsonify({"error": "env not initialized"}), 500

    # 🔥 Rotate tasks (VERY IMPORTANT)
    env = envs[current_env_index % len(envs)]
    current_env_index += 1

    result = env.reset()
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
            "task": current_env_index
        }
    })

# -------- STEP --------
@app.route("/step", methods=["POST"])
def step():
    global env

    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    try:
        data = request.get_json(silent=True) or {}
        action = data.get("action", "support")
    except:
        action = "support"

    result = env.step(action)

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
            "score": float(rew.value) if rew else 0.1
        }
    })

# -------- STATE --------
@app.route("/state", methods=["GET"])
def state():
    global env
    if env is None:
        return jsonify({"error": "env not initialized"}), 500
    return jsonify(env.state())

# -------- MAIN --------
def main():
    print("🚀 RUNNING FLASK ON PORT 7860...")
    app.run(host="0.0.0.0", port=7860, debug=False)

if __name__ == "__main__":
    main()
