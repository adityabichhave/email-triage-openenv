from flask import Flask, request, jsonify
from env import EmailEnv

app = Flask(__name__)
env = EmailEnv()


# -------- RESET --------
<<<<<<< HEAD
@app.route("/reset", methods=["GET", "POST"])
=======
@app.route("/reset", methods=["POST"])
>>>>>>> 00d0977 (final fix: added uv.lock and multi-mode deployment support)
def reset():
    result = env.reset()

    return jsonify({
        "observation": {
            "email": result["observation"].email
<<<<<<< HEAD
        }
=======
        },
        "reward": {
            "value": 0.0
        },
        "done": False,
        "info": {}
>>>>>>> 00d0977 (final fix: added uv.lock and multi-mode deployment support)
    })


# -------- STEP --------
@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True)

    action = data.get("action", None)

    result = env.step(action)

    return jsonify({
<<<<<<< HEAD
        "observation": {
            "email": result["observation"].email
        },
        "reward": {
            "value": result["reward"].value
        },
        "done": result["done"],
        "info": result["info"]
    })
=======
    "observation": {
        "email": result["observation"].email
    },
    "reward": {
        "value": 0.0
    },
    "done": False,
    "info": {}
})
>>>>>>> 00d0977 (final fix: added uv.lock and multi-mode deployment support)


# -------- STATE --------
@app.route("/state", methods=["GET"])
def state():
    return jsonify(env.state())


# -------- ROOT (VERY IMPORTANT) --------
@app.route("/", methods=["GET"])
def home():
    return "OK"


# -------- RUN --------
@app.route("/run", methods=["GET"])
def run():
    return "Email Triage Running 🚀"


if __name__ == "__main__":
<<<<<<< HEAD
    app.run(host="0.0.0.0", port=7860)
=======
    app.run(host="0.0.0.0", port=7860)
>>>>>>> 00d0977 (final fix: added uv.lock and multi-mode deployment support)
