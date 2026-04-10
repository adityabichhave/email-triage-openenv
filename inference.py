import os
from openai import OpenAI

# 1. Start Log
print("[START] task=multi_env env=openenv model=llm", flush=True)

# 2. Config (STRICT)
api_base = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Ensure /v1
if not api_base.endswith("/v1"):
    api_base = f"{api_base}/v1"

client = OpenAI(
    base_url=api_base,
    api_key=api_key
)

# Import all 3 envs
from env.email_env import EmailEnv
from env.sentiment_env import SentimentEnv
from env.priority_env import PriorityEnv


# 🔹 LLM Call
def call_llm(text):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "Classify the email into ONE word: support, sales, complaint, positive, negative, high, or low"
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_tokens=5,
            temperature=0
        )

        content = response.choices[0].message.content.lower()

        # Category
        if "sales" in content: return "sales"
        if "complaint" in content: return "complaint"
        if "support" in content: return "support"

        # Sentiment
        if "positive" in content: return "positive"
        if "negative" in content: return "negative"

        # Priority
        if "high" in content: return "high"
        if "low" in content: return "low"

        return "support"

    except Exception:
        return "support"


# 🔹 Run each environment
def run_env(env):
    rewards = []
    steps = 0

    res = env.reset()
    obs = res["observation"]

    for i in range(1, 10):
        action = call_llm(obs.email)
        result = env.step(action)

        rew = result["reward"].value
        done = result["done"]

        rewards.append(rew)
        steps += 1

        print(f"[STEP] step={steps} action={action} reward={rew:.2f} done={str(done).lower()} error=null", flush=True)

        if done:
            break

        obs = result["observation"]

    return rewards, steps


# 🔹 Main Execution
def main():
    all_rewards = []
    total_steps = 0

    try:
        # Run all 3 tasks (REQUIRED)
        for env_class in [EmailEnv, SentimentEnv, PriorityEnv]:
            env = env_class()
            rewards, steps = run_env(env)

            all_rewards.extend(rewards)
            total_steps += steps

        score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    except Exception as e:
        print(f"[STEP] step={total_steps} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
        score = 0.0

    finally:
        r_str = ",".join(f"{r:.2f}" for r in all_rewards) if all_rewards else "0.00"
        print(f"[END] success={str(score > 0.3).lower()} steps={total_steps} score={score:.2f} rewards={r_str}", flush=True)


if __name__ == "__main__":
    main()
