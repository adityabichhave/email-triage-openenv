class Observation:
    def __init__(self, email):
        self.email = email

class Reward:
    def __init__(self, value):
        self.value = float(value)

class EmailEnv:
    def __init__(self):
        # REQUIRED: Validator uses this list to enumerate and verify 3+ tasks
        self.tasks = [
            {"email": "My order is delayed, please help.", "label": "support"},
            {"email": "I want to buy your product.", "label": "sales"},
            {"email": "I received a damaged item.", "label": "complaint"},
            {"email": "Can you share invoice?", "label": "support"},
            {"email": "Bulk pricing request.", "label": "sales"},
            {"email": "I want a refund.", "label": "complaint"}
        ]
        self.current = 0
        self.email = self.tasks[0]

    def reset(self):
        self.current = 0
        self.email = self.tasks[self.current]
        return {
            "observation": Observation(self.email["email"]),
            "reward": Reward(0.0),
            "done": False,
            "info": {"score": 0.0}
        }

    def step(self, action):
        if self.current >= len(self.tasks):
            return {"observation": Observation("DONE"), "reward": Reward(0.0), "done": True, "info": {"score": 0.0}}

        correct = self.email["label"]
        # Validator strictly requires score in [0.0, 1.0]
        if action == correct:
            reward_val, score_val = 1.0, 1.0
        elif action in ["support", "sales", "complaint"]:
            reward_val, score_val = 0.5, 0.5
        else:
            reward_val, score_val = 0.0, 0.0

        self.current += 1
        done = (self.current >= len(self.tasks))
        
        if not done:
            self.email = self.tasks[self.current]
            next_obs = Observation(self.email["email"])
        else:
            next_obs = Observation("EOF")

        return {
            "observation": next_obs,
            "reward": Reward(float(reward_val)),
            "done": done,
            "info": {"score": float(score_val)}
        }
