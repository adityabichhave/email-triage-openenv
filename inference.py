import os
import sys
from openai import OpenAI
from env.email_env import EmailEnv

# REQUIRED ENVIRONMENT VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def log_start():
    print(f"[START] task=email_triage env=openenv model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success, steps, score, rewards):
    r_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r_str}", flush=True)

def call_llm(text):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Classify as 'support', 'sales', or 'complaint'. One word: {text}"}],
            max_tokens=5
        )
        content = response.choices[0].message.content.strip().lower()
        return next((c for c in ["support", "sales", "complaint"] if c in content), "support")
    except:
        return "support"

def main():
    log_start()
    rewards, steps, score, success = [], 0, 0.0, False
    
    try:
        env = EmailEnv()
        obs_packet = env.reset()
        curr_obs = obs_packet["observation"]

        for i in range(1, 11):
            action = call_llm(curr_obs.email)
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
            success = score >= 0.5
            
    except Exception as e:
        pass
    finally:
        log_end(success, steps, score, rewards)

if __name__ == "__main__":
    main()
