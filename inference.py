import asyncio
import os
from typing import List
from openai import OpenAI

# 🔥 FIX THIS IMPORT BASED ON YOUR STRUCTURE
from server.environment import TaskAction, MultiTaskEnv


API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN")

TASK_NAME = "email-triage"
BENCHMARK = "email-env"
MAX_STEPS = 5


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


# 🔥 REQUIRED: LLM CALL (this fixes your validator error)
def get_label_from_llm(client, email: str) -> str:
    prompt = f"""
Classify this email into one label:
support, sales, positive, negative, high, low

Email: {email}

Return only the label.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().lower()
        return label
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return "support"


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = MultiTaskEnv()

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    all_rewards = []
    total_steps = 0

    try:
        # 🔥 RUN 3 EPISODES (THIS FIXES GRADER ISSUE)
        for episode in range(3):
            result = env.reset()

            email = result.email
            rewards = []

            for step in range(1, MAX_STEPS + 1):
                label = get_label_from_llm(client, email)

                result = env.step(TaskAction(label=label))

                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                total_steps += 1

                log_step(step, label, reward, done)

                if done:
                    break

                email = result.email

            all_rewards.extend(rewards)

        # ✅ FINAL SCORE
        score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        success = score > 0.5

    finally:
        try:
            env.close()
        except:
            pass

        log_end(success, total_steps, score, all_rewards)


if __name__ == "__main__":
    asyncio.run(main())
