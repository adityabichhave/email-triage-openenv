from openenv.core.env_server import Environment
from models import TaskAction, TaskObservation, TaskState


class TaskAction(Action):
    label: str


class TaskObservation(Observation):
    email: str


class TaskState(State):
    pass


class MultiTaskEnv(Environment):  # 🔥 IMPORTANT CHANGE
    def __init__(self):
        self.task_groups = [
            [
                ("Support request: cannot login", "support"),
                ("Need pricing info", "sales"),
                ("Help needed urgently", "support"),
            ],
            [
                ("I love this product", "positive"),
                ("This is terrible", "negative"),
                ("Amazing service", "positive"),
            ],
            [
                ("URGENT issue", "high"),
                ("Can wait", "low"),
                ("Immediate help required", "high"),
            ]
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
        if not self.current_tasks:
            self.reset()

        if self.sample_idx >= len(self.current_tasks):
            return TaskObservation(
                email="",
                done=True,
                reward=0.1
            )

        text, correct = self.current_tasks[self.sample_idx]

        action_label = getattr(action, "label", "").lower().strip()
        correct = correct.lower().strip()

        reward = 0.9 if action_label == correct else 0.1

        self.sample_idx += 1
        done = self.sample_idx >= len(self.current_tasks)

        next_email = ""
        if not done:
            next_email = self.current_tasks[self.sample_idx][0]

        return TaskObservation(
            email=next_email,
            done=done,
            reward=reward
        )

    @property
    def state(self):
        return TaskState()

    def close(self):
        pass
