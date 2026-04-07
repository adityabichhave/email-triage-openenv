import sys
import os

# ✅ Ensure current directory (repo root) is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from env import EmailEnv


def main():
    try:
        env = EmailEnv()

        obs = env.reset()

        actions = ["support", "sales", "complaint"]

        total_score = 0

        for a in actions:
            result = env.step(a)
            total_score += result["info"]["score"]

        print("FINAL SCORE:", total_score / len(actions))

    except Exception as e:
        print("ERROR:", str(e))


if __name__ == "__main__":
    main()
