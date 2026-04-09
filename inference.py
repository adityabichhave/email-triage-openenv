import os
import sys
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import EmailEnv

API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")


def log_start():
    print("[START] task=email_triage env=openenv model=llm", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def get_action(email):
    try:
        url = f"{API_BASE_URL}/chat/completions"

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "Classify email into support, sales, or complaint"},
                {"role": "user", "content": email}
            ],
            "max_tokens": 10
        }

        response = requests.post(url, headers=headers, json=data)

        output = response.json()["choices"][0]["message"]["content"].lower()

        if "sales" in output:
            return "sales"
        elif "complaint" in output:
            return "complaint"
        else:
            return "support"

    except Exception:
        return "support"


def main():
    rewards = []
    steps = 0
    score = 0.0
    success = False

    log_start()

    try:
        env = EmailEnv()
        obs = env.reset()

        for i in range(1, 6):
            email = obs["observation"].email

            action = get_action(email)

            result = env.step(action)

            reward = result.get("reward", {}).get("value", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            steps = i

            log_step(i, action, reward, done)

            obs = result

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)

        success = score > 0

    except Exception as e:
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={str(e)}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    main()
