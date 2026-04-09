import os
import sys
import json
from openai import OpenAI

# Maintain local pathing for email_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env.email_env import EmailEnv
except ImportError:
    from email_env import EmailEnv

# REQUIRED: Pull from environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Initialize OpenAI Client (OpenAI v1.30.1 syntax)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL if API_BASE_URL.endswith("/v1") else f"{API_BASE_URL}/v1"
)

def log_start():
    print("[START] task=email_triage env=openenv model=llm", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def call_llm(email_text):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Classify: support, sales, or complaint. Reply with exactly one word."},
                {"role": "user", "content": email_text}
            ],
            max_tokens=5,
            temperature=0
        )
        content = response.choices[0].message.content.strip().lower()
        # Validation of response categories
        for valid in ["support", "sales", "complaint"]:
            if valid in content:
                return valid
    except Exception:
        pass
    return "support"

def main():
    log_start()
    rewards, steps, score, success = [], 0, 0.0, False
    
    try:
        env = EmailEnv()
        res = env.reset()
        # Handle dict or object return from reset
        obs = res["observation"] if isinstance(res, dict) else res

        for i in range(1, 11):
            # Access the email string from the Observation object
            email = getattr(obs, 'email', "")
            action = call_llm(email)
            
            # Step the environment
            step_res = env.step(action)
            
            # Extract values correctly from step() return
            rew_val = float(step_res["reward"].value)
            is_done = bool(step_res["done"])
            
            rewards.append(rew_val)
            steps = i
            log_step(i, action, rew_val, is_done)

            if is_done:
                break
            obs = step_res["observation"]

        if rewards:
            score = sum(rewards) / len(rewards)
            success = score >= 0.4
            
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error=exception", flush=True)
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
