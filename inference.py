import os
import sys
import json
from openai import OpenAI

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env.email_env import EmailEnv
except ImportError:
    from email_env import EmailEnv

# Configuration from Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Initialize OpenAI Client pointing to the Proxy
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL.rstrip("/") + "/v1" if not API_BASE_URL.endswith("/v1") else API_BASE_URL
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
                {"role": "system", "content": "Classify: support, sales, or complaint. One word only."},
                {"role": "user", "content": email_text}
            ],
            max_tokens=5,
            temperature=0
        )
        content = response.choices[0].message.content.strip().lower()
        for valid in ["support", "sales", "complaint"]:
            if valid in content:
                return valid
    except Exception as e:
        pass
    return "support"

def main():
    log_start()
    rewards, steps, score, success = [], 0, 0.0, False
    
    try:
        env = EmailEnv()
        obs_data = env.reset()
        # Handle both dict and object return types for robustness
        curr_obs = obs_data["observation"] if isinstance(obs_data, dict) else obs_data

        for i in range(1, 11):
            # Extract email string
            email = getattr(curr_obs, 'email', "")
            action = call_llm(email)
            
            res = env.step(action)
            
            rew_val = float(res["reward"].value) if hasattr(res["reward"], 'value') else float(res["reward"])
            is_done = bool(res["done"])
            
            rewards.append(rew_val)
            steps = i
            
            log_step(i, action, rew_val, is_done)

            if is_done:
                break
            curr_obs = res["observation"]

        if rewards:
            score = sum(rewards) / len(rewards)
            success = score >= 0.4
            
    except Exception as e:
        # Avoid crashing so log_end still runs
        print(f"[STEP] step={steps+1} action=none reward=0.0 done=true error=exception", flush=True)
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
