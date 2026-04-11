from openenv.core.env_server import Environment
from models import TaskAction, TaskObservation, TaskState


class MultiTaskEnv(Environment):
    def __init__(self):
        self.task_groups = [
            [("Support request: cannot login", "support")],
            [("I love this product", "positive")],
            [("URGENT issue", "high")]
        ]
        self.current_tasks = []
        self.sample_idx = 0
        self.group_idx = -1

    def reset(self, *args, **kwargs):
        self.group_idx = (self.group_idx + 1) % len(self.task_groups)
        self.current_tasks = self.task_groups[self.group_idx]
        self.sample_idx = 0

        return TaskObservation(
            email=self.current_tasks[0][0],
            done=False,
            reward=0.1
        )

    def step(self, action: TaskAction, **kwargs):
        text, correct = self.current_tasks[self.sample_idx]

        action_label = action.label.lower().strip()
        reward = 0.9 if action_label == correct else 0.1

        self.sample_idx += 1
        done = True  # single-step task → IMPORTANT

        return TaskObservation(
            email="",
            done=done,
            reward=reward
        )

    @property
    def state(self):
        return TaskState()

    def close(self):
        pass
