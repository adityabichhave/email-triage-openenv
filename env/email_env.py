class Observation:
    def __init__(self, email, prompt=None, messages=None):
        self.email = email
        self.prompt = prompt or f"Classify this email: {email}"
        self.messages = messages or []


class Reward:
    def __init__(self, value):
        self.value = float(value)


class EmailEnv:
    def __init__(self):
        self.tasks = [
            {"email": "Support request: I cannot login.", "label": "support"},
            {"email": "Sales inquiry: Pricing for 500 units?", "label": "sales"},
            {"email": "Complaint: My order arrived broken.", "label": "complaint"},
            {"email": "Hello, I want to buy a subscription.", "label": "sales"},
            {"email": "I am angry about the shipping delay.", "label": "complaint"}
        ]
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

    def reset(self):
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

        return {
            "observation": Observation(self.email["email"]),
            "reward": Reward(0.1),  # ✅ valid
            "done": False,
            "info": {"score": 0.1}  # ✅ valid
        }

    def step(self, action):
        target = self.email["label"]
        action = str(action).strip().lower()

        if action == target:
            score_val = 0.9
        else:
            score_val = 0.2

        self.current_idx += 1
        done = self.current_idx >= len(self.tasks)

        if not done:
            self.email = self.tasks[self.current_idx]
            next_email = self.email["email"]
        else:
            next_email = "DONE"

        return {
            "observation": Observation(next_email),
            "reward": Reward(score_val),
            "done": done,
            "info": {"score": float(score_val)}
        }
