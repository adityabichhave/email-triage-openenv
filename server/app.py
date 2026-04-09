import sys
import os
import importlib.util
from flask import Flask, request, jsonify

app = Flask(__name__)
env = None

def load_env_manually():
    """Manually loads the EmailEnv class from the relative path."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_file = os.path.join(base_dir, "env", "email_env.py")
    
    if not os.path.exists(target_file):
        target_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_env.py")

    try:
        spec = importlib.util.spec_from_file_location("email_env", target_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.EmailEnv()
    except Exception as e:
        print(f"❌ ENV LOAD FAILED: {e}")
        return None

# --- API ENDPOINTS ---

@app.route("/", methods=["GET"])
def home():
    return "OK"

@app.route("/reset", methods=["POST"])
def reset():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500
    res = env.reset()
    return jsonify({
        "observation": {"email": getattr(res["observation"], 'email', "")},
        "reward": {"value": 0.0},
        "done": False,
        "info": res.get("info", {"score": 0.05})
    })

@app.route("/step", methods=["POST"])
def step():
    if env is None:
        return jsonify({"error": "env not initialized"}), 500
    data = request.get_json(force=True)
    action = data.get("action", "support")
    result = env.step(action)
    return jsonify({
        "observation": {"email": getattr(result["observation"], 'email', "")},
        "reward": {"value": float(result["reward"].value)},
        "done": bool(result["done"]),
        "info": result.get("info", {"score": 0.5})
    })

@app.route("/state", methods=["GET"])
def state():
    if env is None: return jsonify({"error": "env not initialized"}), 500
    return jsonify(env.state())

# --- VALIDATOR REQUIREMENTS ---

def main():
    """
    Explicit main() function called by the validator or entry point.
    """
    global env
    print("🔥 APP STARTING via main()...")
    env = load_env_manually()
    if env:
        print("✅ ENV LOADED SUCCESSFULLY")
    app.run(host="0.0.0.0", port=7860, debug=False)

if __name__ == "__main__":
    main()
