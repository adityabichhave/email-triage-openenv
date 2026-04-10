from openenv.core.env_server import create_fastapi_app
from server.environment import MultiTaskEnv, TaskAction, TaskObservation

# OpenEnv app
app = create_fastapi_app(
    MultiTaskEnv,
    TaskAction,
    TaskObservation
)

# 🔥 REQUIRED FOR VALIDATOR
def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )

# 🔥 REQUIRED ENTRYPOINT
if __name__ == "__main__":
    main()
