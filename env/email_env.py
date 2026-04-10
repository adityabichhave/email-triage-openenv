from pydantic import BaseModel
from typing import List, Optional


class Observation(BaseModel):
    email: str
    prompt: Optional[str] = None
    messages: Optional[List[str]] = []


class Reward(BaseModel):
    value: float


class EmailEnv:
    def __init__(self):
        self.tasks = [
            {"email": "Support request: I cannot login.", "label": "support"},
            {"email": "Sales inquiry: Pricing for 500 units?", "label": "sales"},
            {"email": "Complaint: My order arrived broken.", "label": "complaint"},
            {"email": "I want to buy a subscription.", "label": "sales"},
            {"email": "Shipping delay made me angry.", "label": "complaint"},
        ]
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

    def reset(self):
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

        return {
            "observation": Observation(email=self.email["email"]),   # ✅ FIXED
            "reward": Reward(value=0.0),                             # ✅ FIXED
            "done": False,
            "info": {"score": 0.0}
        }

    def step(self, action):
        target = self.email["label"]
        action_str = str(action).strip().lower()

        score_val = 1.0 if action_str == target else 0.0

        self.current_idx += 1
        done = self.current_idx >= len(self.tasks)

        if not done:
            self.email = self.tasks[self.current_idx]

        return {
            "observation": Observation(email=self.email["email"]),   # ✅ FIXED
            "reward": Reward(value=float(score_val)),                # ✅ FIXED
            "done": done,
            "info": {"score": float(score_val)}
        }

    def state(self):
        return {
            "current_idx": self.current_idx,
            "email": self.email["email"]
        }
