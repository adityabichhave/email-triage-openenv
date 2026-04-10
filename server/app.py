from openenv.core.env_server import create_fastapi_app

# ❌ WRONG (remove this)
# from ..models import EmailAction, EmailObservation
# from .environment import MultiTaskEnv

# ✅ CORRECT
from models import EmailAction, EmailObservation
from server.environment import MultiTaskEnv

app = create_fastapi_app(
    MultiTaskEnv,
    EmailAction,
    EmailObservation
)
