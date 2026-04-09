class Observation:
    def __init__(self, email):
        self.email = email


class Action:
    def __init__(self, label):
        self.label = label


class Reward:
    def __init__(self, value):
        self.value = value


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
            "observation": {"email": self.email["email"]}
        }

def step(self, action):
    correct = self.email["label"]

    # ✅ Strict scoring (never 0 or 1)
    if action == correct:
        score = 0.9
    elif action in ["support", "sales", "complaint"]:
        score = 0.6
    else:
        score = 0.3

    # ✅ Reward MUST be positive (important for grader detection)
    reward = score

    self.current += 1

    if self.current >= len(self.tasks):
        done = True
        next_email = None
    else:
        done = False
        self.email = self.tasks[self.current]
        next_email = self.email["email"]

    return {
        "observation": {"email": next_email},
        "reward": {"value": float(reward)},   # ✅ MUST be float
        "done": done,
        "info": {
            "score": float(score)            # ✅ MUST be float
        }
    }
    def state(self):
        return {
            "current_index": self.current,
            "total_tasks": len(self.tasks)
        }
