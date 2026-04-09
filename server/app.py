import sys
import os
import importlib.util
from flask import Flask, request, jsonify

print("🔥 APP STARTING...")

# --- FAIL-SAFE LOADER ---
def load_env_manually():
    # 1. Try to find email_env.py by looking relative to this server file
    # Path: /app/server/app.py -> /app/env/email_env.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, "env", "email_env.py")
    
    # 2. Fallback: Check if it's in the same directory (for flat structures)
    if not os.path.exists(target_file):
        target_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_env.py")

    try:
        spec = importlib.util.spec_from_file_location("email_env", target_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.EmailEnv()
    except Exception as e:
        print(f"❌ MANUAL LOAD FAILED at {target_file}: {e}")
        return None

env = load_env_manually()

if env:
    print("✅ ENV LOADED SUCCESSFULLY VIA DIRECT PATH")
else:
    print("❌ SYSTEM ALERT: ENV STILL NOT INITIALIZED")

app = Flask(__name__)

@app.route("/reset", methods=["POST"])
def reset():
    if env is None:
        return jsonify({"error": "env not initialized", "tried_path": "check logs"}), 500
    
    res = env.reset()
    return jsonify({
        "observation": {"email": getattr(res["observation"], 'email', "")},
        "reward": {"value": 0.0},
        "done": False,
        "info": {"score": 0.05}
    })

@app.route("/step", methods=["POST"])
def step():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    data = request.get_json(force=True)
    action = data.get("action", "support")
    
    result = env.step(action)
    
    # Ensure nested JSON structure matches OpenEnv spec
    return jsonify({
        "observation": {
            "email": getattr(result["observation"], 'email', "")
        },
        "reward": {
            "value": float(result["reward"].value)
        },
        "done": bool(result["done"]),
        "info": result.get("info", {"score": 0.5})
    })

@app.route("/state", methods=["GET"])
def state():
    if env is None: return jsonify({"error": "env not initialized"}), 500
    return jsonify(env.state())

@app.route("/", methods=["GET"])
def home():
    return "OK"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
