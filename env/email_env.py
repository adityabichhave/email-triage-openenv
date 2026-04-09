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
        # Ensure reset structure matches the expected observation format
        return {
            "observation": Observation(self.email["email"]),
            "reward": Reward(0.0),
            "done": False,
            "info": {"score": 0.05} # Non-zero start
        }

    def step(self, action):
        if self.email is None:
            self.reset()

        correct = self.email["label"]
        # Validator strict range fix: use 0.95 and 0.05 instead of 1.0/0.0
        if action == correct:
            reward, score = 0.9, 0.95
        elif action in ["support", "sales", "complaint"]:
            reward, score = 0.5, 0.5
        else:
            reward, score = 0.1, 0.1

        self.current += 1
        done = self.current >= len(self.tasks)
        
        if not done:
            self.email = self.tasks[self.current]
            next_email = self.email["email"]
        else:
            next_email = "DONE" # Safety string

        return {
            "observation": Observation(next_email),
            "reward": Reward(float(reward)),
            "done": done,
            "info": {"score": float(score)}
        }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
