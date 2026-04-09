import os
import sys
import json
import urllib.request
import importlib.util

# 1. DYNAMIC LOADER (Fixes ModuleNotFoundError)
def get_env_class():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Check root or /env/ subfolder
    possible_paths = [
        os.path.join(base_path, "email_env.py"),
        os.path.join(base_path, "env", "email_env.py")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("email_env", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.EmailEnv
    raise ImportError("email_env.py not found in workspace")

# 2. LOGGING (Mandatory Format)
def log_start():
    print("[START] task=email_triage env=openenv model=llm", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

# 3. API CALL (Strict LiteLLM Proxy Usage)
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")

def call_llm(text):
    url = f"{API_BASE_URL}/chat/completions"
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": f"Classify: support, sales, or complaint. Reply one word only: {text}"}],
        "temperature": 0
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), 
                                   headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as res:
            resp = json.loads(res.read().decode())
            out = resp["choices"][0]["message"]["content"].strip().lower()
            return next((c for c in ["support", "sales", "complaint"] if c in out), "support")
    except:
        return "support"

# 4. MAIN INFERENCE LOOP
def main():
    log_start()
    rewards, steps, score, success = [], 0, 0.0, False
    
    try:
        EmailEnv = get_env_class()
        env = EmailEnv()
        obs_packet = env.reset()
        curr_obs = obs_packet["observation"]

        # Run until the environment says 'done'
        for i in range(1, 11):
            email_text = getattr(curr_obs, 'email', "")
            action = call_llm(email_text)
            
            res = env.step(action)
            
            # Extract data for logging
            rew_val = float(res["reward"].value)
            is_done = bool(res["done"])
            rewards.append(rew_val)
            steps = i
            
            # Log step so validator can count it as a "graded task"
            log_step(i, action, rew_val, is_done)

            if is_done:
                break
            curr_obs = res["observation"]

        if rewards:
            score = sum(rewards) / len(rewards)
            success = score >= 0.4
            
    except Exception as e:
        # Error logging that won't break the [START]/[END] requirement
        print(f"[STEP] step={steps+1} action=none reward=0.0 done=true error={str(e).replace(' ', '_')}", flush=True)
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
