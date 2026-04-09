import os
import sys
import json
from openai import OpenAI

# 1. IMMEDIATE LOGGING
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. PROXY CONFIGURATION - THE KEY FIX
# We must ensure the base_url points exactly to the v1 completions endpoint
raw_base_url = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "")

# LiteLLM Proxy expects the base_url to end with /v1 for the OpenAI SDK
if raw_base_url and not raw_base_url.endswith("/v1"):
    formatted_base_url = f"{raw_base_url}/v1"
else:
    formatted_base_url = raw_base_url

# Initialize Client with explicit environment variables
client = OpenAI(
    api_key=api_key,
    base_url=formatted_base_url
)

# 3. ENVIRONMENT IMPORT
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env.email_env import EmailEnv
except ImportError:
    from email_env import EmailEnv

def call_llm(text):
    """Makes the actual API call that the validator MUST see."""
    try:
        # We use the model_name provided by the validator
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Classify as: support, sales, or complaint. One word only."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=10
        )
        # Extract response
        res_text = response.choices[0].message.content.strip().lower()
        for category in ["support", "sales", "complaint"]:
            if category in res_text:
                return category
    except Exception as e:
        # If the API fails, we log it to help you debug
        print(f"DEBUG: API Call failed: {e}")
    return "support"

def main():
    rewards, steps, score, success = [], 0, 0.0, False
    try:
        env = EmailEnv()
        obs_packet = env.reset()
        # Handle observation object
        curr_obs = obs_packet["observation"] if isinstance(obs_packet, dict) else obs_packet

        # Execute exactly 6 tasks (from your email_env.py)
        for i in range(1, 7):
            email_body = getattr(curr_obs, 'email', "")
            
            # --- THE API CALL HAPPENS HERE ---
            action = call_llm(email_body)
            
            res = env.step(action)
            
            # Extract reward value (assuming Reward object has .value)
            rew_val = float(res["reward"].value)
            is_done = bool(res["done"])
            
            rewards.append(rew_val)
            steps = i
            
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(is_done).lower()} error=null", flush=True)

            if is_done:
                break
            curr_obs = res["observation"]

        if rewards:
            score = sum(rewards) / len(rewards)
            success = score >= 0.4
            
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
