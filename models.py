from openenv.core.env_server import Action, Observation, State
from typing import Optional


class EmailAction(Action):
    label: str


class EmailObservation(Observation):
    email: str


class EmailState(State):
    task_type: Optional[str] = None
