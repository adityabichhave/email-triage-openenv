import os
import sys
from openai import OpenAI

# 1. Start logging immediately
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. Extract Environment Variables (Injected by Validator)
api_base = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# CRITICAL PROXY FIX: 
# The OpenAI SDK requires the '/v1' suffix to route chat completions properly.
# Without this, the proxy won't register your calls.
if api_base and not api_base.endswith("/v1"):
    formatted_base_url = f"{api_base}/v1"
else:
    formatted_base_url = api_base

# Initialize Client strictly using the validator's credentials
client = OpenAI(
    api_key=api_key,
    base_url=formatted_base_url
)

# 3. Load your environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.email_env import EmailEnv

def call_llm(text):
    """Hits the LiteLLM proxy to record API calls."""
    try:
        # You MUST use the model_name variable from the environment
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Classify: support, sales, or complaint. One word only."},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=10
        )
        content = response.choices[0].message.content.strip().lower()
        return next((c for c in ["support", "sales", "complaint"] if c in content), "support")
    except Exception as e:
        # If API fails, we log a hint for the participant log
        print(f"DEBUG: Proxy call failed: {e}", flush=True)
        return "support"

def main():
    rewards, steps, score = [], 0, 0.0
    try:
        env = EmailEnv()
        res = env.reset()
        obs = res["observation"] if isinstance(res, dict) else res

        # Loop through tasks (Make sure to call the API for every task)
        for i in range(1, 11):
            email_body = getattr(obs, 'email', "")
            
            # --- THIS CALL MUST REGISTER ON THE PROXY ---
            action = call_llm(email_body)
            
            step_res = env.step(action)
            
            # Extract reward value (assumes Reward object has .value)
            rew_val = float(step_res["reward"].value) if hasattr(step_res["reward"], 'value') else float(step_res["reward"])
            is_done = bool(step_res["done"])
            
            rewards.append(rew_val)
            steps = i
            
            print(f"[STEP] step={i} action={action} reward={rew_val:.2f} done={str(is_done).lower()} error=null", flush=True)

            if is_done:
                break
            obs = step_res["observation"]

        score = sum(rewards) / len(rewards) if rewards else 0.0
            
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(score >= 0.4).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
