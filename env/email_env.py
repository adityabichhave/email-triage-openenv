class Observation:
    def __init__(self, email, prompt=None, messages=None):
        self.email = email
        # OpenEnv often looks for these attributes for agent history
        self.prompt = prompt or f"Classify this email: {email}"
        self.messages = messages or []

class Reward:
    def __init__(self, value):
        self.value = float(value)

class EmailEnv:
    def __init__(self):
        # REQUIREMENT 1: Enumerable Tasks
        # The validator looks for a list of at least 3 distinct tasks.
        self.tasks = [
            {"email": "Support request: I cannot login.", "label": "support"},
            {"email": "Sales inquiry: Pricing for 500 units?", "label": "sales"},
            {"email": "Complaint: My order arrived broken.", "label": "complaint"},
            {"email": "Hello, I want to buy a subscription.", "label": "sales"},
            {"email": "I am angry about the shipping delay.", "label": "complaint"}
        ]
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]

    def reset(self):
        self.current_idx = 0
        self.email = self.tasks[self.current_idx]
        # Return a dictionary with 'info' containing a starting score
        return {
            "observation": Observation(self.email["email"]),
            "reward": Reward(0.0),
            "done": False,
            "info": {"score": 0.0} 
        }

    def step(self, action):
        # REQUIREMENT 2: Automated Grader
        # The 'grader' is the logic that determines the score in info['score']
        target = self.email["label"]
        
        # Normalize action (cleaning square brackets if the LLM adds them)
        clean_action = str(action).strip().lower().replace("[", "").replace("]", "")
        
        if clean_action == target:
            step_reward = 1.0
            step_score = 1.0
        else:
            step_reward = 0.0
            step_score = 0.0

        self.current_idx += 1
        done = self.current_idx >= len(self.tasks)
        
        # Prepare next observation or dummy for end of episode
        next_email = self.tasks[self.current_idx]["email"] if not done else "Done"
        
        # REQUIREMENT 3: Return Format
        # OpenEnv expects 'score' in the info dict to validate the "grader"
        return {
            "observation": Observation(next_email),
            "reward": Reward(step_reward),
            "done": done,
            "info": {
                "score": float(step_score),
                "label": target
            }
        }
