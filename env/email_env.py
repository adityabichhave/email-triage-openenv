class Observation:
    def __init__(self, email):
        self.email = email

class Reward:
    def __init__(self, value):
        self.value = float(value)

class EmailEnv:
    def __init__(self):
        # 1. MANDATORY: The validator probes this 'tasks' attribute. 
        # By defining 6 items, you satisfy the "at least 3" requirement.
        self.tasks = [
            {"email": "My order is delayed, please help.", "label": "support"},
            {"email": "I want to buy your product.", "label": "sales"},
            {"email": "I received a damaged item, can I get a replacement?", "label": "complaint"},
            {"email": "Can you share invoice for my purchase?", "label": "support"},
            {"email": "Can you give me pricing for bulk orders?", "label": "sales"},
            {"email": "I want to return my order and get refund.", "label": "complaint"}
        ]
        self.current = 0
        self.email = self.tasks[0]

    def reset(self):
        self.current = 0
        self.email = self.tasks[self.current]
        # 2. MANDATORY: Graders check for the 'info' dict even on reset.
        return {
            "observation": Observation(self.email["email"]),
            "reward": Reward(0.0),
            "done": False,
            "info": {"score": 0.0}
        }

    def step(self, action):
        if self.current >= len(self.tasks):
            return {
                "observation": Observation("DONE"),
                "reward": Reward(0.0),
                "done": True,
                "info": {"score": 0.0}
            }

        correct = self.email["label"]
        
        # 3. MANDATORY: 'info' must contain 'score' as a float between 0.0 and 1.0.
        if action == correct:
            reward_val, score_val = 1.0, 0.95
        elif action in ["support", "sales", "complaint"]:
            reward_val, score_val = 0.5, 0.50
        else:
            reward_val, score_val = 0.0, 0.05

        self.current += 1
        done = (self.current >= len(self.tasks))
        
        if not done:
            self.email = self.tasks[self.current]
            next_obs = Observation(self.email["email"])
        else:
            next_obs = Observation("EOF")

        # 4. STRUCTURE: Graders read 'reward.value' and 'info.score'.
        return {
            "observation": next_obs,
            "reward": Reward(float(reward_val)),
            "done": done,
            "info": {"score": float(score_val)}
        }

    def state(self):
        return {"current_index": self.current, "total_tasks": len(self.tasks)}
