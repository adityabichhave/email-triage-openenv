import os
import sys
import json

# Log Start immediately to satisfy the "No structured output" check
print("[START] task=email_triage env=openenv model=llm", flush=True)

try:
    from openai import OpenAI
except ImportError:
    print("[STEP] step=0 action=none reward=0.00 done=true error=ModuleNotFoundError_openai", flush=True)
    print("[END] success=false steps=0 score=0.00 rewards=0.00", flush=True)
    sys.exit(0) # Exit gracefully so validator parses the error step

# Maintain local pathing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env.email_env import EmailEnv
except ImportError:
    from email_env import EmailEnv

# Config
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# Initialize Client - LiteLLM Proxy requires /v1 suffix usually
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL if API_BASE_URL.endswith("/v1") else f"{API_BASE_URL}/v1"
)

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
            if valid in content: return valid
    except:
        pass
    return "support"

def main():
    rewards, steps, score, success = [], 0, 0.0, False
    try:
        env = EmailEnv()
        res = env.reset()
        # Handle observation wrapper
        obs = res["observation"] if isinstance(res, dict) else res

        for i in range(1, 11):
            email = getattr(obs, 'email', "No content")
            action = call_llm(email)
            
            step_res = env.step(action)
            
            # Extracting float values safely
            rew_val = float(step_res["reward"].value)
            is_done = bool(step_res["done"])
            
            rewards.append(rew_val)
            steps = i
            
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(is_done).lower()} error=null", flush=True)

            if is_done: break
            obs = step_res["observation"]

        if rewards:
            score = sum(rewards) / len(rewards)
            success = score >= 0.4
            
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={str(e).replace(' ', '_')}", flush=True)
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    main()
