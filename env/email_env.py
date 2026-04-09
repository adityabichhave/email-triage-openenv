from openenv import Observation, Reward


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
            "observation": Observation(email=self.email["email"])
        }

    def step(self, action):
        correct = self.email["label"]

        # ✅ STRICT RANGE (0,1)
        if action == correct:
            score = 0.9
        elif action in ["support", "sales", "complaint"]:
            score = 0.6
        else:
            score = 0.3

        reward = score

        self.current += 1

        if self.current >= len(self.tasks):
            done = True
            next_email = ""
        else:
            done = False
            self.email = self.tasks[self.current]
            next_email = self.email["email"]

        return {
            "observation": Observation(email=next_email),
            "reward": Reward(value=float(reward)),   # ✅ REAL OPENENV OBJECT
            "done": done,
            "info": {"score": float(score)}
        }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
