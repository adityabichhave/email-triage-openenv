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

        # ✅ STRICT VALID RANGE
        if action == correct:
            reward = 0.8
            score = 0.9
        elif action in ["support", "sales", "complaint"]:
            reward = 0.5
            score = 0.6
        else:
            reward = 0.2
            score = 0.3

        self.current += 1

        # ✅ STOP NATURALLY (NO RESET, NO LOOP)
        if self.current >= len(self.tasks):
            done = True
        else:
            done = False
            self.email = self.tasks[self.current]

        return {
            "observation": Observation(self.email["email"] if not done else ""),
            "reward": Reward(reward),
            "done": done,
            "info": {"score": score}
        }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
