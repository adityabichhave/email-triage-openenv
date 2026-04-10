from typing import Optional
from openenv.core.env_server import Action, Observation, State


class EmailAction(Action):
    label: str


class EmailObservation(Observation):
    email: str


class EmailState(State):
    task_type: str = ""
