from pydantic import BaseModel

# --------- Models ---------
class Observation(BaseModel):
    email: str

class Action(BaseModel):
    label: str  # support / sales / complaint


# --------- Environment ---------
class EmailEnv:
    def __init__(self):
        self.tasks = {
            "easy": [
                {"email": "My order is delayed, please help.", "label": "support"},
                {"email": "I want to buy your product.", "label": "sales"}
            ],
            "medium": [
                {"email": "I received a damaged item, can I get a replacement?", "label": "complaint"},
                {"email": "Can you give me pricing for bulk orders?", "label": "sales"}
            ],
            "hard": [
                {"email": "My product is late and I might cancel and buy from somewhere else.", "label": "complaint"},
                {"email": "I want to return my order but also interested in buying another one.", "label": "complaint"}
            ]
        }

        self.levels = ["easy", "medium", "hard"]
        self.current_level = 0
        self.current_index = 0

    def reset(self):
        self.current_level = 0
        self.current_index = 0

        level = self.levels[self.current_level]
        self.current_task = self.tasks[level][self.current_index]

        return {
            "observation": Observation(email=self.current_task["email"])
        }

    def step(self, action: Action):
        correct = self.current_task["label"]

        if action.label == correct:
            reward = 1.0
            score = 1.0
        elif action.label in ["support", "sales", "complaint"]:
            reward = 0.2
            score = 0.5
        else:
            reward = -1.0
            score = 0.0

        self.current_index += 1

        level = self.levels[self.current_level]

        if self.current_index >= len(self.tasks[level]):
            self.current_level += 1
            self.current_index = 0

        if self.current_level >= len(self.levels):
            done = True
            next_email = None
        else:
            level = self.levels[self.current_level]
            self.current_task = self.tasks[level][self.current_index]
            next_email = self.current_task["email"]
            done = False

        return {
            "observation": Observation(email=next_email) if next_email else None,
            "reward": reward,
            "done": done,
            "info": {"score": score}
        }

    def state(self):
        return {
            "level": self.current_level,
            "index": self.current_index
        }