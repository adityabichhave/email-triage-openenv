from openenv.core.env_server import Action, Observation, State


class TaskAction(Action):
    label: str


class TaskObservation(Observation):
    email: str


class TaskState(State):
    pass


class MultiTaskEnv:
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

        self.group_idx = -1
        self.sample_idx = 0
        self.current_tasks = []

    # ✅ SAFE RESET (NO HASH, NO GLOBAL, NO CRASH)
    def reset(self, episode_id=None, seed=None):
        self.group_idx = (self.group_idx + 1) % len(self.task_groups)
        self.current_tasks = self.task_groups[self.group_idx]
        self.sample_idx = 0

        return TaskObservation(
            email=self.current_tasks[0][0],
            done=False,
            reward=0.1
        )

    async def reset_async(self, *args, **kwargs):
        return self.reset(*args, **kwargs)

    # ✅ SAFE STEP
    def step(self, action: TaskAction):
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

    async def step_async(self, action):
        return self.step(action)

    @property
    def state(self):
        return {"group": self.group_idx}

    # ✅ REQUIRED by OpenEnv (fixes your crash)
    def close(self):
        pass
