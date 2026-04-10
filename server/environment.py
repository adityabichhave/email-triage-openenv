from openenv.core.env_server import Action, Observation, State


class TaskAction(Action):
    label: str


class TaskObservation(Observation):
    email: str


class TaskState(State):
    task_type: str = ""


class MultiTaskEnv:
    def __init__(self):
        self.task_groups = [
            ("email", [
                ("Support request: cannot login", "support"),
                ("Need pricing info", "sales"),
            ]),
            ("sentiment", [
                ("I love this product", "positive"),
                ("This is terrible", "negative"),
            ]),
            ("priority", [
                ("URGENT issue", "high"),
                ("Can wait", "low"),
            ])
        ]

        self.group_index = -1
        self.current_tasks = []
        self.i = 0
        self._state = TaskState()

    def reset(self):
        # 🔥 IMPORTANT: rotate task each reset
        self.group_index = (self.group_index + 1) % 3

        task_type, tasks = self.task_groups[self.group_index]
        self.current_tasks = tasks
        self.i = 0

        # 🔥 THIS LINE MAKES VALIDATOR SEE TASKS
        self._state = TaskState(task_type=task_type)

        text, _ = self.current_tasks[self.i]

        return TaskObservation(
            email=text,
            done=False,
            reward=0.1
        )

    def step(self, action: TaskAction):
        text, correct = self.current_tasks[self.i]

        label = getattr(action, "label", "")
        score = 0.9 if label == correct else 0.1

        self.i += 1
        done = self.i >= len(self.current_tasks)

        next_text = ""
        if not done:
            next_text = self.current_tasks[self.i][0]

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
