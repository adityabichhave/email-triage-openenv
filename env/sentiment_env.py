from pydantic import BaseModel

class Observation(BaseModel):
    email: str

class Reward(BaseModel):
    value: float

class SentimentEnv:
    def __init__(self):
        self.tasks = [
            ("I love this service", "positive"),
            ("This is terrible", "negative"),
            ("Very happy with product", "positive"),
        ]
        self.i = 0

    def reset(self):
        self.i = 0
        return {"observation": Observation(email=self.tasks[0][0]), "reward": Reward(0.0), "done": False, "info": {}}

    def step(self, action):
        correct = self.tasks[self.i][1]
        score = 1.0 if action == correct else 0.0

        self.i += 1
        done = self.i >= len(self.tasks)

        obs = self.tasks[self.i][0] if not done else ""

        return {
            "observation": Observation(email=obs),
            "reward": Reward(score),
            "done": done,
            "info": {"score": score}
        }

    def state(self):
        return {"index": self.i}
