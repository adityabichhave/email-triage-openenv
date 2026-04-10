from pydantic import BaseModel

class Observation(BaseModel):
    email: str

class Reward(BaseModel):
    value: float

class EmailEnv:
    def __init__(self):
        self.tasks = [
            ("I cannot login", "support"),
            ("Pricing details?", "sales"),
            ("Product is broken", "complaint"),
        ]
        self.i = 0

    def reset(self):
        self.i = 0
        return {
            "observation": Observation(email=self.tasks[0][0]),
            "reward": Reward(value=0.1),
            "done": False,
            "info": {"score": 0.1}
        }

    def step(self, action):
        correct = self.tasks[self.i][1]
        score = 0.9 if action == correct else 0.1

        self.i += 1
        done = self.i >= len(self.tasks)

        next_email = ""
        if not done:
            next_email = self.tasks[self.i][0]

        return {
            "observation": Observation(email=next_email),
            "reward": Reward(value=score),
            "done": done,
            "info": {"score": score}
        }

    def state(self):
        return {"index": self.i}
