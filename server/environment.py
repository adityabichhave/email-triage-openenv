from openenv.core.env_server import Environment
from models import TaskAction, TaskObservation, TaskState


class MultiTaskEnv(Environment):
    def __init__(self):
        # ✅ all rewards strictly between (0,1)
        self.task_groups = [
            ("Support request: cannot login", "support", 0.8),
            ("I love this product", "positive", 0.6),
            ("URGENT issue", "high", 0.7),
        ]
        self.group_idx = -1

    def reset(self, *args, **kwargs):
        self.group_idx = (self.group_idx + 1) % len(self.task_groups)

        text, _, _ = self.task_groups[self.group_idx]

        return TaskObservation(
            email=text,
            done=False,
            reward=0.5   # ✅ NOT 0
        )

    def step(self, action: TaskAction, **kwargs):
        text, correct, base_reward = self.task_groups[self.group_idx]

        action_label = action.label.lower().strip()

        if action_label == correct:
            reward = base_reward      # e.g. 0.6 / 0.7 / 0.8
        else:
            reward = 0.2              # ✅ NOT 0

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


# 🔥 GRADERS (ALSO MUST FOLLOW RANGE)

def grade_support(action, observation):
    return 0.8 if action.label.lower() == "support" else 0.2


def grade_sentiment(action, observation):
    return 0.7 if action.label.lower() == "positive" else 0.2


def grade_priority(action, observation):
    return 0.9 if action.label.lower() == "high" else 0.3
