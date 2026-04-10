from openenv.core.env_server import Action, Observation, State


class TaskAction(Action):
    label: str


class TaskObservation(Observation):
    email: str


class TaskState(State):
    pass


class MultiTaskEnv:
    def __init__(self):
        # 🔥 3 DIFFERENT TASK TYPES
        self.tasks = [
            ("I cannot login", "support"),
            ("I love this product", "positive"),
            ("URGENT issue", "high"),
        ]
        self.i = 0
        self._state = TaskState()

    def reset(self):
        self.i = 0
        text, _ = self.tasks[self.i]

        return TaskObservation(
            email=text,
            done=False,
            reward=0.1
        )

    def step(self, action: TaskAction):
        text, correct = self.tasks[self.i]

        action_label = getattr(action, "label", "")
        score = 0.9 if action_label == correct else 0.1

        self.i += 1
        done = self.i >= len(self.tasks)

        next_text = ""
        if not done:
            next_text = self.tasks[self.i][0]

        return TaskObservation(
            email=next_text,
            done=done,
            reward=score
        )

    @property
    def state(self):
        return self._state

    def close(self):
        pass
