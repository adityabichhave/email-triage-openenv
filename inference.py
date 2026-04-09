import os
import sys
import json
import urllib.request

# Ensure the local environment can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from email_env import EmailEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-3.5-turbo")

def log_start():
    # MUST be the first thing printed
    print("[START] task=email_triage env=openenv model=llm", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def call_llm(email):
    try:
        url = f"{API_BASE_URL.rstrip('/')}/chat/completions"
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "Reply ONLY with one word: support, sales, or complaint"},
                {"role": "user", "content": email}
            ],
            "max_tokens": 5
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
        text = result["choices"][0]["message"]["content"].strip().lower()
        for choice in ["support", "sales", "complaint"]:
            if choice in text: return choice
    except:
        pass
    return "support"

def main():
    # 1. Initialize variables before the TRY block
    rewards = []
    steps = 0
    score = 0.0
    success = False

    # 2. Start logging immediately
    log_start()

    try:
        env = EmailEnv()
        # Initial reset gives us the first email
        init_res = env.reset()
        current_obs = init_res["observation"]

        # Loop through tasks
        for i in range(1, 10):
            action = call_llm(current_obs.email)
            
            # Step the environment
            step_result = env.step(action)
            
            reward = float(step_result["reward"].value)
            done = bool(step_result["done"])
            
            rewards.append(reward)
            steps = i

            # 3. Log the step IMMEDIATELY
            log_step(i, action, reward, done)

            if done:
                break
            
            current_obs = step_result["observation"]

        if len(rewards) > 0:
            score = sum(rewards) / len(rewards)
            success = score >= 0.5

    except Exception as e:
        # If it fails, we still need to print a step to avoid "No steps found"
        print(f"[STEP] step={steps+1} action=error reward=0.00 done=true error=exception", flush=True)
    
    finally:
        # 4. End logging
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
