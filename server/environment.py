from openenv.core.env_server import Environment
from models import TaskAction, TaskObservation, TaskState


class MultiTaskEnv(Environment):
    def __init__(self):
        self.task_groups = [
            # TASK 1 → classification
            ("Support request: cannot login", "support", 1.0),

            # TASK 2 → sentiment
            ("I love this product", "positive", 0.5),

            # TASK 3 → priority
            ("URGENT issue", "high", 0.8),
        ]
        self.group_idx = -1

    def reset(self, *args, **kwargs):
        self.group_idx = (self.group_idx + 1) % len(self.task_groups)

        text, _, _ = self.task_groups[self.group_idx]

        return TaskObservation(
            email=text,
            done=False,
            reward=0.0
        )

    def step(self, action: TaskAction, **kwargs):
        text, correct, base_reward = self.task_groups[self.group_idx]

        action_label = action.label.lower().strip()

        if action_label == correct:
            reward = base_reward   # 🔥 DIFFERENT REWARD PER TASK
        else:
            reward = 0.0

        return TaskObservation(
            email="",
            done=True,
            reward=reward
        )

    @property
    def state(self):
        return TaskState()

    def close(self):
        pass
