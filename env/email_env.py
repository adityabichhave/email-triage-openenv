class Observation:
    def __init__(self, email):
        self.email = email


class EmailEnv:
    def __init__(self):
        self.tasks = [
            {"email": "My order is delayed, please help.", "label": "support"},
            {"email": "I want to buy your product.", "label": "sales"},
            {"email": "I received a damaged item, can I get a replacement?", "label": "complaint"},
            {"email": "Can you share invoice for my purchase?", "label": "support"},
            {"email": "Can you give me pricing for bulk orders?", "label": "sales"},
            {"email": "I want to return my order and get refund.", "label": "complaint"}
        ]
        self.current = 0
        self.email = None

    def reset(self):
        self.current = 0
        self.email = self.tasks[self.current]

        return {
            "observation": Observation(self.email["email"])
        }

def step(self, action):
    if self.email is None:
        self.reset()

    correct = self.email["label"]

    if action == correct:
        reward = 0.8
        score = 0.9
    elif action in ["support", "sales", "complaint"]:
        reward = 0.5
        score = 0.6
    else:
        reward = 0.2
        score = 0.3

    # ✅ MOVE UPDATE AFTER COMPUTATION
    self.current += 1

    if self.current >= len(self.tasks):
        self.current = 0
        done = False
    else:
        done = False

    # 🔥 CRITICAL: ALWAYS UPDATE EMAIL AFTER CURRENT IS FINAL
    self.email = self.tasks[self.current]

    return {
        "observation": Observation(self.email["email"]),
        "reward": Reward(float(reward)),
        "done": done,
        "info": {"score": float(score)}
    }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
