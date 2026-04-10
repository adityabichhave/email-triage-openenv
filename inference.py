import os
import sys
from openai import OpenAI

# 1. MANDATORY LOGGING
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. PROXY SETTINGS
raw_url = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Ensure the URL is perfect for the proxy
base_url = f"{raw_url}/v1" if not raw_url.endswith("/v1") else raw_url

# 3. INITIALIZE AI
client = OpenAI(api_key=api_key, base_url=base_url)

# 4. LOAD THE ENVIRONMENT
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.email_env import EmailEnv

def main():
    rewards, steps = [], 0
    try:
        env = EmailEnv()
        res = env.reset()
        obs = res["observation"]

        # Loop through all tasks in the environment
        for i in range(1, 11):
            # Ask the AI to classify the email
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Classify as 'support', 'sales', or 'complaint'. One word only: {obs.email}"}],
                temperature=0,
                max_tokens=5
            )
            action = response.choices[0].message.content.strip().lower()

            # Tell the environment what the AI said
            res = env.step(action)
            rew_val = float(res["reward"].value)
            
            rewards.append(rew_val)
            steps = i
            
            # Print progress for the validator
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(res['done']).lower()} error=null", flush=True)

            if res["done"]:
                break
            obs = res["observation"]

    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
    finally:
        # Final Summary
        final_score = sum(rewards) / len(rewards) if rewards else 0.0
        r_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(final_score >= 0.4).lower()} steps={steps} score={final_score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
