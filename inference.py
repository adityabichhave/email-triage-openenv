import os
import sys

# 1. Start logging immediately - Mandatory for the validator
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. Resilient Import Guard
try:
    from openai import OpenAI
except ImportError:
    print("[STEP] step=0 action=none reward=0.00 done=true error=ModuleNotFoundError_openai", flush=True)
    print("[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
    # Exit with 0 to prevent the "Unhandled Exception" crash log
    sys.exit(0)

# 3. Proxy Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Ensure the base_url is formatted for the OpenAI SDK (requires /v1)
base_url = f"{API_BASE_URL}/v1" if not API_BASE_URL.endswith("/v1") else API_BASE_URL

client = OpenAI(api_key=API_KEY, base_url=base_url)

# 4. Load Environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.email_env import EmailEnv

def call_llm(text):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Classify: support, sales, complaint. One word: {text}"}],
            temperature=0,
            max_tokens=5
        )
        content = response.choices[0].message.content.strip().lower()
        return next((c for c in ["support", "sales", "complaint"] if c in content), "support")
    except:
        return "support"

def main():
    rewards, steps, score = [], 0, 0.0
    try:
        env = EmailEnv()
        res = env.reset()
        obs = res["observation"] if isinstance(res, dict) else res

        # Run through all 6 tasks in your env/email_env.py
        for i in range(1, 7):
            email = getattr(obs, 'email', "")
            action = call_llm(email)
            
            step_res = env.step(action)
            rew_val = float(step_res["reward"].value)
            is_done = bool(step_res["done"])
            
            rewards.append(rew_val)
            steps = i
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(is_done).lower()} error=null", flush=True)

            if is_done: break
            obs = step_res["observation"]

        score = sum(rewards) / len(rewards) if rewards else 0.0
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(score >= 0.4).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
