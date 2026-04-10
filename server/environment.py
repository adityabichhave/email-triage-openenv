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
            ("email", [
                ("Support request: cannot login", "support"),
                ("Need pricing info", "sales"),
                ("Help needed urgently", "support"),
            ]),
            ("sentiment", [
                ("I love this product", "positive"),
                ("This is terrible", "negative"),
                ("Amazing service", "positive"),
            ]),
            ("priority", [
                ("URGENT issue", "high"),
                ("Can wait", "low"),
                ("Immediate help required", "high"),
            ])
        ]

        self.group_idx = -1
        self.sample_idx = 0
        self.current_tasks = []
        self._state = TaskState()

    # 🔥 RESET (NO info field)
    def reset(self, *args, **kwargs):
        self.group_idx = (self.group_idx + 1) % len(self.task_groups)

        task_type, self.current_tasks = self.task_groups[self.group_idx]
        self.sample_idx = 0

        text, _ = self.current_tasks[self.sample_idx]

        print("RESET TASK:", task_type)  # debug log

        return TaskObservation(
            email=text,
            done=False,
            reward=0.1
        )

    async def reset_async(self, *args, **kwargs):
        return self.reset(*args, **kwargs)

    # 🔥 STEP (reward acts as score)
    def step(self, action: TaskAction):
        text, correct = self.current_tasks[self.sample_idx]

        action_label = getattr(action, "label", "").lower().strip()
        correct = correct.lower().strip()

        reward = 0.9 if action_label == correct else 0.1

        self.sample_idx += 1
        done = self.sample_idx >= len(self.current_tasks)

        next_text = ""
        if not done:
            next_text = self.current_tasks[self.sample_idx][0]

        return TaskObservation(
            email=next_text,
            done=done,
            reward=reward
        )

    async def step_async(self, action):
        return self.step(action)

    @property
    def state(self):
        return self._state

    def close(self):
        pass
