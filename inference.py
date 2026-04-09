import os
import sys
import json
import urllib.request
import importlib.util

# --- FORCE IMPORT OF email_env.py ---
def load_env():
    # Search in root and /env/ directory
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "env")
    ]
    
    for path in search_paths:
        file_path = os.path.join(path, "email_env.py")
        if os.path.exists(file_path):
            spec = importlib.util.spec_from_file_location("email_env", file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["email_env"] = module
            spec.loader.exec_module(module)
            return module.EmailEnv
    raise ImportError("Could not locate email_env.py in root or /env/")

try:
    EmailEnv = load_env()
except Exception as e:
    # We define a fallback class ONLY so log_start() can execute and you can see errors
    class EmailEnv:
        def __init__(self): self.tasks = [{"email": "err", "label": "err"}]
        def reset(self): return {"observation": type('O', (), {'email': ''}), "reward": type('R', (), {'value': 0.0}), "done": False, "info": {"score": 0.0}}
        def step(self, a): return self.reset()

# --- CONFIGURATION ---
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")

def log_start():
    print("[START] task=email_triage env=openenv model=llm", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def call_llm(email_text):
    url = f"{API_BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": "Reply: support, sales, or complaint. One word."},
                    {"role": "user", "content": email_text}],
        "max_tokens": 5
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), 
                                   headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode())
            return res["choices"][0]["message"]["content"].strip().lower()
    except: return "support"

def main():
    log_start() # Must be first!
    rewards, steps, score, success = [], 0, 0.0, False
    
    try:
        env = EmailEnv()
        obs_data = env.reset()
        # Use getattr to safely handle the Observation object
        curr_obs = obs_data["observation"]

        for i in range(1, 10):
            email = getattr(curr_obs, 'email', "")
            action = call_llm(email)
            
            res = env.step(action)
            reward = float(res["reward"].value)
            done = bool(res["done"])
            
            rewards.append(reward)
            steps = i
            log_step(i, action, reward, done)

            if done: break
            curr_obs = res["observation"]

        if rewards:
            score = sum(rewards) / len(rewards)
            success = score > 0.4
    except Exception as e:
        print(f"[STEP] step={steps+1} action=none reward=0.0 done=true error={str(e).replace(' ', '_')}", flush=True)
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
