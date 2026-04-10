from openenv.core.env_server import Action, Observation, State


class TaskAction(Action):
    label: str


class TaskObservation(Observation):
    email: str


class TaskState(State):
    task_type: str = ""


class MultiTaskEnv:
    def __init__(self):
        # 🔥 3 DISTINCT TASK GROUPS
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
        self.i = 0
        self.current_group = None
        self._state = TaskState()

    def reset(self):
        # 🔥 SWITCH TASK GROUP EACH RESET (CRITICAL)
        self.group_index = (self.group_index + 1) % 3
        task_type, tasks = self.task_groups[self.group_index]

        self.current_group = tasks
        self.i = 0

        # 🔥 SET TASK TYPE
        self._state = TaskState(task_type=task_type)

        text, _ = self.current_group[self.i]

        return TaskObservation(
            email=text,
            done=False,
            reward=0.1
        )

    def step(self, action: TaskAction):
        text, correct = self.current_group[self.i]

        action_label = getattr(action, "label", "")
        score = 0.9 if action_label == correct else 0.1

        self.i += 1
        done = self.i >= len(self.current_group)

        next_text = ""
        if not done:
            next_text = self.current_group[self.i][0]

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
