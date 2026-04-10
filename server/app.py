import sys
import os
import importlib.util
from flask import Flask, request, jsonify

app = Flask(__name__)
env = None


def load_env_manually():
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
        print(f"❌ ENV LOAD FAILED: {e}", flush=True)
        return None


# --- API ENDPOINTS ---

@app.route("/", methods=["GET"])
def home():
    return "OK"


@app.route("/reset", methods=["POST"])
def reset():
    global env

    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    try:
        print("🔄 RESET CALLED", flush=True)

        res = env.reset()
        obs = res["observation"]

        return jsonify({
            "observation": {
                "email": obs.email
            },
            "reward": {
                "value": float(res["reward"].value)
            },
            "done": bool(res["done"]),
            "info": res.get("info", {"score": 0.0})
        })

    except Exception as e:
        print(f"❌ RESET ERROR: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/step", methods=["POST"])
def step():
    global env

    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    try:
        data = request.get_json(force=True)
        action = data.get("action", "support")

        print(f"➡️ STEP CALLED with action={action}", flush=True)

        result = env.step(action)
        obs = result["observation"]

        return jsonify({
            "observation": {
                "email": obs.email
            },
            "reward": {
                "value": float(result["reward"].value)
            },
            "done": bool(result["done"]),
            "info": result.get("info", {"score": 0.0})
        })

    except Exception as e:
        print(f"❌ STEP ERROR: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/state", methods=["GET"])
def state():
    global env

    if env is None:
        return jsonify({"error": "env not initialized"}), 500

    try:
        return jsonify(env.state())
    except Exception as e:
        print(f"❌ STATE ERROR: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


# --- ENTRY POINT ---

def main():
    global env

    print("🔥 APP STARTING via main()...", flush=True)

    env = load_env_manually()

    if env is None:
        print("❌ ENV FAILED TO LOAD - EXITING", flush=True)
        raise Exception("Env failed to load")

    print("✅ ENV LOADED SUCCESSFULLY", flush=True)

    app.run(host="0.0.0.0", port=7860, debug=False)


if __name__ == "__main__":
    main()
