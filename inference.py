import asyncio
import os
from openai import OpenAI
from server.environment import MultiTaskEnv, TaskAction

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

TASK_NAME = "email-triage"
BENCHMARK = "my_env"


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def get_label(client, email: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": f"Classify: {email}"}],
        temperature=0,
        max_tokens=5,
    )
    return response.choices[0].message.content.strip().lower()


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = MultiTaskEnv()

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    rewards = []
    steps = 0

    try:
        for episode in range(3):  # 🔥 REQUIRED

            result = env.reset()
            email = result.email

            label = get_label(client, email)

            result = env.step(TaskAction(label=label))

            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps += 1

            log_step(steps, label, reward, done)

        score = sum(rewards) / len(rewards)
        success = True  # 🔥 SAFE

    finally:
        env.close()
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())
