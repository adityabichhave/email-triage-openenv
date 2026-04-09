import os
import sys
from openai import OpenAI

# 1. Start logging immediately
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. Extract Environment Variables
# The validator injects these: DO NOT hardcode them.
api_base = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# 3. THE CRITICAL PROXY FIX
# OpenAI SDK v1.x expects the base_url to include the /v1 suffix.
# LiteLLM proxies will NOT track your calls if this is missing.
if api_base and not api_base.endswith("/v1"):
    api_base = f"{api_base}/v1"

# Initialize the OpenAI Client
client = OpenAI(
    api_key=api_key,
    base_url=api_base
)

# 4. Import your local environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.email_env import EmailEnv

def call_llm(text):
    """Hits the proxy using the validator's credentials."""
    try:
        response = client.chat.completions.create(
            model=model_name, # MUST use the model_name from environment
            messages=[
                {"role": "system", "content": "Classify: support, sales, or complaint. One word only."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().lower()
        return next((c for c in ["support", "sales", "complaint"] if c in content), "support")
    except Exception:
        return "support"

def main():
    rewards, steps, score = [], 0, 0.0
    try:
        env = EmailEnv()
        obs_data = env.reset()
        obs = obs_data["observation"] if isinstance(obs_data, dict) else obs_data

        # Loop through tasks (Make sure you call the API for every task)
        for i in range(1, 11):
            email_body = getattr(obs, 'email', "")
            
            # --- THIS CALL REGISTERS ON THE PROXY ---
            action = call_llm(email_body)
            
            res = env.step(action)
            rew_val = float(res["reward"].value)
            is_done = bool(res["done"])
            
            rewards.append(rew_val)
            steps = i
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(is_done).lower()} error=null", flush=True)

            if is_done:
                break
            obs = res["observation"]

        score = sum(rewards) / len(rewards) if rewards else 0.0
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(score >= 0.4).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
