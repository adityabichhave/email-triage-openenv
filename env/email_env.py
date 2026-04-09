class Observation:
    def __init__(self, email):
        self.email = email


class Reward:
    def __init__(self, value):
        self.value = float(value)


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
        reward_val = 0.7
        score_val = 0.9
    elif action in ["support", "sales", "complaint"]:
        reward_val = 0.5
        score_val = 0.6
    else:
        reward_val = 0.3
        score_val = 0.4

    self.current += 1

    # 🔥 CRITICAL FIX
    if self.current >= len(self.tasks):
        self.current = 0
        self.email = self.tasks[self.current]
        done = False   # ✅ NEVER TRUE
    else:
        done = False
        self.email = self.tasks[self.current]

    return {
        "observation": Observation(self.email["email"]),
        "reward": Reward(float(reward_val)),
        "done": done,
        "info": {"score": float(score_val)}
    }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
