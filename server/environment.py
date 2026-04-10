import random
from typing import Dict
from ..models import EmailAction, EmailObservation, EmailState


class MultiTaskEnv:
    def __init__(self):
        self.tasks = [
            ("I cannot login", "support", "email"),
            ("Pricing details?", "sales", "email"),
            ("Product broken", "complaint", "email"),

            ("I love this", "positive", "sentiment"),
            ("This is bad", "negative", "sentiment"),

            ("URGENT issue", "high", "priority"),
            ("Check later", "low", "priority"),
        ]
        self.i = 0
        self.state = EmailState()

    def reset(self) -> EmailObservation:
        self.i = 0
        email, _, task = self.tasks[self.i]

        self.state = EmailState(task_type=task)

        return EmailObservation(
            email=email,
            done=False,
            reward=0.1
        )

    def step(self, action: EmailAction) -> EmailObservation:
        email, correct, task = self.tasks[self.i]

        score = 0.9 if action.label == correct else 0.1

        self.i += 1
        done = self.i >= len(self.tasks)

        next_email = ""
        if not done:
            next_email = self.tasks[self.i][0]

        return EmailObservation(
            email=next_email,
            done=done,
            reward=score
        )
