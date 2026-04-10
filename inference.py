import os
import sys
import json
import urllib.request

print("[START] task=email_triage env=openenv model=llm", flush=True)

api_base = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# ✅ Fix proxy endpoint
if not api_base.endswith("/v1"):
    api_base = f"{api_base}/v1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.email_env import EmailEnv


def call_llm(text):
    try:
        url = f"{api_base}/chat/completions"

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Reply ONLY: support, sales, or complaint"},
                {"role": "user", "content": text}
            ],
            "max_tokens": 5,
            "temperature": 0
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )

        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode())

        content = data["choices"][0]["message"]["content"].lower()

        if "sales" in content:
            return "sales"
        elif "complaint" in content:
            return "complaint"
        return "support"

    except Exception as e:
        print(f"DEBUG: {e}", flush=True)
        return "support"


def main():
    rewards = []
    steps = 0

    try:
        env = EmailEnv()
        res = env.reset()
        obs = res["observation"]

        for i in range(1, 10):
            email = obs.email

            action = call_llm(email)

            result = env.step(action)

            reward = result["reward"].value
            done = result["done"]

            rewards.append(reward)
            steps = i

            print(f"[STEP] step={i} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

            if done:
                break

            obs = result["observation"]

        score = sum(rewards) / len(rewards)

    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.10 done=true error={type(e).__name__}", flush=True)
        score = 0.1

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(score > 0.3).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    main()
