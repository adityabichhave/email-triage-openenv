from pydantic import BaseModel


# ----------- Models -----------

class Observation(BaseModel):
    email: str | None


class Action(BaseModel):
    label: str


class Reward(BaseModel):
    value: float


# ----------- Agent Logic -----------

def agent(email: str) -> str:
    email = email.lower()

    if "buy" in email or "price" in email or "pricing" in email or "order" in email:
        return "sales"

    elif "refund" in email or "return" in email or "damaged" in email or "issue" in email:
        return "complaint"

    else:
        return "support"


# ----------- Environment -----------

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

    def step(self, action=None):
        email_text = self.email["email"]

        predicted_label = agent(email_text)
        action = Action(label=predicted_label)

        correct = self.email["label"]

        if action.label == correct:
            reward = 1.0
            score = 1.0
        elif action.label in ["support", "sales", "complaint"]:
            reward = 0.2
            score = 0.5
        else:
            reward = -1.0
            score = 0.0

        self.current += 1

        if self.current >= len(self.tasks):
            done = True
            next_email = None
        else:
            done = False
            self.email = self.tasks[self.current]
            next_email = self.email["email"]

        return {
            "observation": Observation(email=next_email),
            "reward": Reward(value=reward),
            "done": done,
            "info": {
                "score": score,
                "correct_label": correct,
                "predicted_label": action.label
            }
        }

    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
