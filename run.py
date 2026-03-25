from env import EmailEnv, Action

env = EmailEnv()

obs = env.reset()
print("START:", obs)

actions = ["support", "sales", "complaint", "sales", "complaint", "complaint"]

for a in actions:
    result = env.step(Action(label=a))
    print("ACTION:", a, "→", result)