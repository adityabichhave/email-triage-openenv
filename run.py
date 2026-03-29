from flask import Flask, request, jsonify
from env import EmailEnv

app = Flask(__name__)
env = EmailEnv()


# -------- RESET --------
@app.route("/reset", methods=["GET", "POST"])
def reset():
    result = env.reset()

    return jsonify({
        "observation": {
            "email": result["observation"].email
        }
    })


# -------- STEP --------
@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True)

    action = data.get("action", None)

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
    app.run(host="0.0.0.0", port=7860)
