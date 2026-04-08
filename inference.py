import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import EmailEnv


def log_start():
    print("[START] task=email_triage env=openenv model=rule_based", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def main():
    rewards = []
    steps = 0
    success = False
    score = 0.0

    log_start()

    try:
        env = EmailEnv()
        obs = env.reset()

        actions = ["support", "sales", "complaint"]

        for i, action in enumerate(actions, start=1):
            result = env.step(action)

            reward = result.get("reward", {}).get("value", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            steps = i

            log_step(i, action, reward, done)

            if done:
                break

        if rewards:
            score = sum(rewards) / len(rewards)

        success = score > 0

    except Exception as e:
        print(f"[STEP] step=0 action=none reward=0.00 done=true error={str(e)}", flush=True)

    finally:
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    main()
