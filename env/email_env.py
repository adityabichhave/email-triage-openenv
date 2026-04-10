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
        # 3 TYPES OF TASKS (Graders)
        self.tasks = [
            # --- TASK 1: CATEGORY ---
            {"email": "I cannot login to my account.", "type": "category", "label": "support"},
            {"email": "Pricing for bulk order?", "type": "category", "label": "sales"},

            # --- TASK 2: SENTIMENT ---
            {"email": "I am very angry about this delay.", "type": "sentiment", "label": "negative"},
            {"email": "Great service, thank you!", "type": "sentiment", "label": "positive"},

            # --- TASK 3: PRIORITY ---
            {"email": "URGENT: system is down!", "type": "priority", "label": "high"},
            {"email": "Whenever possible, check this issue.", "type": "priority", "label": "low"},
        ]

        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

    def reset(self):
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

        return {
            "observation": Observation(email=self.email["email"]),
            "reward": Reward(value=0.0),
            "done": False,
            "info": {"score": 0.0}
        }

    def step(self, action):
        target = self.email["label"]
        task_type = self.email["type"]

        action_str = str(action).strip().lower()

        # --- GRADER LOGIC ---
        if action_str == target:
            score_val = 1.0
        else:
            score_val = 0.2

        self.current_idx += 1
        done = self.current_idx >= len(self.tasks)

        if not done:
            self.email = self.tasks[self.current_idx]

        return {
            "observation": Observation(email=self.email["email"]),
            "reward": Reward(value=float(score_val)),
            "done": done,
            "info": {
                "score": float(score_val),
                "task_type": task_type
            }
        }

    def state(self):
        return {
            "current_idx": self.current_idx,
            "email": self.email["email"]
        }
