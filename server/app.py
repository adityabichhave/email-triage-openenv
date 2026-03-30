from flask import Flask, request, jsonify
from env import EmailEnv

app = Flask(__name__)
env = EmailEnv()


@app.route("/reset", methods=["POST"])
def reset():
    result = env.reset()
    return jsonify({
        "observation": {"email": result["observation"].email},
        "reward": {"value": 0.0},
        "done": False,
        "info": {}
    })


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True)
    action = data.get("action")

    result = env.step(action)

    return jsonify({
        "observation": {"email": result["observation"].email},
        "reward": {"value": result["reward"].value},
        "done": result["done"],
        "info": result["info"]
    })


@app.route("/state", methods=["GET"])
def state():
    return jsonify(env.state())


@app.route("/", methods=["GET"])
def home():
    return "OK"