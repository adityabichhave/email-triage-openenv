from openenv.core.env_server import create_fastapi_app
from ..models import EmailAction, EmailObservation
from .environment import MultiTaskEnv

app = create_fastapi_app(
    MultiTaskEnv,
    EmailAction,
    EmailObservation
)
