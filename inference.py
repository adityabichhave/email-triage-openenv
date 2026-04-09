import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import EmailEnv
from openai import OpenAI


# 🔥 REQUIRED ENV VARIABLES
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]


# 🔥 MANDATORY OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)


# -------- LOGS --------
def log_start():
    print("[START] task=email_triage env=openenv model=llm", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# -------- LLM CALL --------
def call_llm(email):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply ONLY with one word: support, sales, or complaint"},
            {"role": "user", "content": email}
        ],
        max_tokens=5,
    )

    text = response.choices[0].message.content.strip().lower()

    # 🔥 HARD FILTER (guarantee valid action)
    if "sales" in text:
        return "sales"
    elif "complaint" in text:
        return "complaint"
    else:
        return "support"


# -------- MAIN --------
def main():
    rewards = []
    steps = 0

    log_start()

    try:
        env = EmailEnv()
        obs = env.reset()

        # 🔥 FIXED NUMBER OF STEPS (NO BREAK)
        for i in range(1, 6):
            email = obs["observation"].email

            action = call_llm(email)

            result = env.step(action)

            reward = result["reward"].value
            done = result["done"]

            rewards.append(reward)
            steps = i

            log_step(i, action, reward, done)

            obs = result

        score = sum(rewards) / len(rewards)
        success = score > 0

    except Exception as e:
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={str(e)}", flush=True)
        success = False
        score = 0.0

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    main()
