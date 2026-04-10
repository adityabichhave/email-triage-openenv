from models import EmailAction, EmailObservation, EmailState


class MultiTaskEnv:
    def __init__(self):
        self.tasks = [
            ("I cannot login", "support"),
            ("Pricing details?", "sales"),
            ("Product broken", "complaint"),
            ("I love this", "positive"),
            ("This is bad", "negative"),
            ("URGENT issue", "high"),
            ("Check later", "low"),
        ]
        self.i = 0

    def reset(self):
        self.i = 0
        email, _ = self.tasks[self.i]

        return EmailObservation(
            email=email,
            done=False,
            reward=0.1
        )

    def step(self, action: EmailAction):
        email, correct = self.tasks[self.i]

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

    @property
    def state(self):
        return {}

    # ✅ ADD THIS
    def close(self):
        pass
