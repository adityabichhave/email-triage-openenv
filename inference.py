import os
import sys
import json
import urllib.request
import importlib.util

# --- PATH FIX: Find env/email_env.py relative to this script ---
def get_env_class():
    base_path = os.path.dirname(os.path.abspath(__file__))
    # Look specifically in the 'env' folder as per your structure
    env_path = os.path.join(base_path, "env", "email_env.py")
    
    if not os.path.exists(env_path):
        # Fallback to root if not in env/
        env_path = os.path.join(base_path, "email_env.py")

    spec = importlib.util.spec_from_file_location("email_env", env_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EmailEnv

# --- LOGGING ---
def log_start():
    print("[START] task=email_triage env=openenv model=llm", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

# --- API ---
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")

def call_llm(text):
    url = f"{API_BASE_URL}/chat/completions"
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": f"Classify email: support, sales, or complaint. One word: {text}"}],
        "temperature": 0
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode(), 
                                   headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as res:
            resp = json.loads(res.read().decode())
            content = resp["choices"][0]["message"]["content"].strip().lower()
            return next((c for c in ["support", "sales", "complaint"] if c in content), "support")
    except:
        return "support"

# --- MAIN ---
def main():
    log_start()
    rewards, steps, score, success = [], 0, 0.0, False
    
    try:
        EmailEnv = get_env_class()
        env = EmailEnv()
        obs_packet = env.reset()
        curr_obs = obs_packet["observation"]

        for i in range(1, 10):
            email_text = getattr(curr_obs, 'email', "")
            action = call_llm(email_text)
            
            res = env.step(action)
            rew_val = float(res["reward"].value)
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
        print(f"[STEP] step={steps+1} action=none reward=0.0 done=true error={str(e).replace(' ', '_')}", flush=True)
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
