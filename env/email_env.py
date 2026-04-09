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
        correct = self.email["label"]

        # ✅ STRICT (0,1)
        if action == correct:
            reward = 0.7
            score = 0.9
        elif action in ["support", "sales", "complaint"]:
            reward = 0.5
            score = 0.6
        else:
            reward = 0.3
            score = 0.4

        self.current += 1

        if self.current >= len(self.tasks):
            done = True
            next_email = ""
        else:
            done = False
            self.email = self.tasks[self.current]
            next_email = self.email["email"]

        return {
            "observation": Observation(next_email),   # ✅ OBJECT
            "reward": Reward(reward),                 # ✅ OBJECT
            "done": done,
            "info": {"score": float(score)}           # ✅ FLOAT
        }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
