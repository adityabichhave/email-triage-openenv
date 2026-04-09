import os
import sys
import json
from openai import OpenAI

# 1. IMMEDIATE LOGGING FOR PARSER
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. PROXY CONFIGURATION (The Critical Part)
# We pull variables injected by the validator environment
api_base = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "")

# Most LiteLLM proxies require the /v1 suffix for the OpenAI SDK to work properly.
# We ensure the URL is formatted exactly as the proxy expects.
if api_base and not api_base.endswith("/v1"):
    formatted_url = f"{api_base}/v1"
else:
    formatted_url = api_base

# Initialize Client using ONLY the environment variables
client = OpenAI(
    api_key=api_key,
    base_url=formatted_url
)

# 3. ENVIRONMENT IMPORT
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env.email_env import EmailEnv
except ImportError:
    # Fallback for different directory structures
    from email_env import EmailEnv

def call_llm(text):
    """Hits the LiteLLM Proxy using the provided credentials."""
    try:
        # Mandatory: Use model_name from environment variable
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Classify as: support, sales, or complaint. One word only."},
                {"role": "user", "content": text}
            ],
            temperature=0, # Deterministic for grading
            max_tokens=10
        )
        res_text = response.choices[0].message.content.strip().lower()
        # Map response to valid categories
        for category in ["support", "sales", "complaint"]:
            if category in res_text:
                return category
    except Exception as e:
        # Silently fail to 'support' to keep the loop running
        pass
    return "support"

def main():
    rewards, steps, score = [], 0, 0.0
    try:
        env = EmailEnv()
        obs_packet = env.reset()
        # Handle observation wrapper
        curr_obs = obs_packet["observation"] if isinstance(obs_packet, dict) else obs_packet

        # The env has 6 tasks; we loop through all of them
        for i in range(1, 11):
            email_body = getattr(curr_obs, 'email', "")
            
            # --- TRIGGER PROXY API CALL ---
            action = call_llm(email_body)
            
            res = env.step(action)
            
            # Extract reward value and check if done
            rew_val = float(res["reward"].value)
            is_done = bool(res["done"])
            
            rewards.append(rew_val)
            steps = i
            
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(is_done).lower()} error=null", flush=True)

            if is_done:
                break
            curr_obs = res["observation"]

        score = sum(rewards) / len(rewards) if rewards else 0.0
            
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(score >= 0.4).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
