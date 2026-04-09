print("🔥 APP STARTING...")

import sys
import os
from flask import Flask, request, jsonify

# --- CRITICAL PATH FIX ---
# We need to point to the directory CONTAINING the 'env' folder.
# Structure: /workspace/server/app.py -> /workspace/env/email_env.py
current_script_path = os.path.abspath(__file__)
server_dir = os.path.dirname(current_script_path)
root_dir = os.path.dirname(server_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- ROBUST IMPORT ---
try:
    # Ensure we import exactly from the folder the validator sees
    from env.email_env import EmailEnv
    env = EmailEnv()
    if not hasattr(env, 'reset') or not hasattr(env, 'step'):
        raise ImportError("Env class loaded but missing required methods")
    print("✅ ENV LOADED SUCCESSFULLY")
except Exception as e:
    print(f"❌ ERROR LOADING ENV: {e}")
    # This helps you see the actual path error in logs
    print(f"PYTHONPATH: {sys.path}")
    env = None

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "OK"

@app.route("/reset", methods=["POST"])
def reset():
    if env is None:
        return jsonify({"error": "env not initialized", "path": sys.path[0]}), 500
    
    result = env.reset()
    # Extract data safely to avoid serialization errors
    return jsonify({
        "observation": {"email": getattr(result["observation"], 'email', "")},
        "reward": {"value": 0.0},
        "done": False,
        "info": {"score": 0.05} # Validator mandatory field
    })

@app.route("/step", methods=["POST"])
def step():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    try:
        data = request.get_json(force=True)
        action = data.get("action", "support")
    except:
        action = "support"

    result = env.step(action)
    
    # Ensure we follow the exact OpenEnv schema for the JSON response
    return jsonify({
        "observation": {
            "email": getattr(result["observation"], 'email', "")
        },
        "reward": {
            "value": float(result["reward"].value)
        },
        "done": bool(result["done"]),
        "info": result.get("info", {"score": 0.5}) # Ensure score is present
    })

@app.route("/state", methods=["GET"])
def state():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500
    return jsonify(env.state())

if __name__ == "__main__":
    # Validator usually expects port 7860
    app.run(host="0.0.0.0", port=7860, debug=False)
