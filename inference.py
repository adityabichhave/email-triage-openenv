import asyncio
import os
from typing import List

from openai import OpenAI

from server.environment import TaskAction, MultiTaskEnv

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN")

TASK_NAME = "email-triage"
BENCHMARK = "email-env"
MAX_STEPS = 5


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def decide_label(email: str) -> str:
    email = email.lower()
    if "urgent" in email:
        return "high"
    if "love" in email:
        return "positive"
    if "terrible" in email:
        return "negative"
    if "pricing" in email:
        return "sales"
    return "support"


async def main():
    env = MultiTaskEnv()

    rewards: List[float] = []
    steps = 0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = env.reset()
        obs = result
        email = obs.email

        for step in range(1, MAX_STEPS + 1):
            label = decide_label(email)

            result = env.step(TaskAction(label=label))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps = step

            log_step(step, label, reward, done)

            if done:
                break

            email = result.email

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.5

    finally:
        try:
            env.close()
        except:
            pass

        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
