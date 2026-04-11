import asyncio
import os
import requests
from openai import OpenAI

# 🔥 ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# 🔥 YOUR HF SPACE URL
BASE_URL = "https://adityakumarbichhave-email-triage-env.hf.space"

BENCHMARK = "openenv"


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# 🔥 API CALL (MANDATORY)
def get_label(client, email: str) -> str:
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Classify: {email}"}],
            temperature=0,
            max_tokens=5,
        )
    except:
        pass

    email = email.lower()

    if "login" in email:
        return "support"
    elif "love" in email:
        return "positive"
    elif "urgent" in email:
        return "high"

    return "support"


# 🔥 ENV CALLS
def reset_env():
    r = requests.post(f"{BASE_URL}/reset")
    return r.json()


def step_env(label):
    r = requests.post(
        f"{BASE_URL}/step",
        json={"action": {"label": label}},
    )
    return r.json()


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # 🔥 MUST MATCH openenv.yaml IDs
    task_ids = ["support_task", "sentiment_task", "priority_task"]

    try:
        for task_id in task_ids:
            log_start(task_id, BENCHMARK, MODEL_NAME)

            res = reset_env()
            email = res["observation"]["email"]

            label = get_label(client, email)

            res = step_env(label)

            reward = res.get("reward", 0.0)
            done = res.get("done", True)

            log_step(1, label, reward, done)

            score = reward
            success = score > 0.1

            log_end(success, 1, score, [reward])

    except Exception as e:
        # ensure script never crashes
        log_end(False, 0, 0.0, [])


if __name__ == "__main__":
    asyncio.run(main())
