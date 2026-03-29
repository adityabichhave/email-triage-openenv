from env import EmailEnv

env = EmailEnv()

obs = env.reset()

actions = ["support", "sales", "complaint"]

total_score = 0

for a in actions:
    result = env.step(a)
    total_score += result["info"]["score"]

print("FINAL SCORE:", total_score / len(actions))
