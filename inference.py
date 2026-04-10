import os
import sys
import json
import urllib.request

# 1. Start Log
print("[START] task=email_triage env=openenv model=llm", flush=True)

# 2. Config
api_base = os.environ.get("API_BASE_URL", "").rstrip("/")
api_key = os.environ.get("API_KEY", "")
model_name = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

if not api_base.endswith("/v1"):
    api_base = f"{api_base}/v1"

# Load local env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env.email_env import EmailEnv
except ImportError:
    from email_env import EmailEnv

def call_llm(text):
    try:
        url = f"{api_base}/chat/completions"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Reply ONLY with one word: support, sales, or complaint"},
                {"role": "user", "content": text}
            ],
            "max_tokens": 5,
            "temperature": 0
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read().decode())
        content = data["choices"][0]["message"]["content"].lower()
        
        if "sales" in content: return "sales"
        if "complaint" in content: return "complaint"
        return "support"
    except Exception:
        return "support"

def main():
    rewards = []
    steps = 0
    try:
        env = EmailEnv()
        res = env.reset()
        obs = res["observation"]

        # Loop through all 5 tasks
        for i in range(1, 6):
            action = call_llm(obs.email)
            result = env.step(action)
            
            rew = result["reward"].value
            done = result["done"]
            rewards.append(rew)
            steps = i

            print(f"[STEP] step={i} action={action} reward={rew:.2f} done={str(done).lower()} error=null", flush=True)
            
            if done: break
            obs = result["observation"]

        score = sum(rewards) / len(rewards) if rewards else 0.0
    except Exception as e:
        steps += 1
        print(f"[STEP] step={steps} action=none reward=0.00 done=true error={type(e).__name__}", flush=True)
        score = 0.0
    finally:
        r_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(score > 0.3).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

if __name__ == "__main__":
    main()
