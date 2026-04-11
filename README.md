# 🚀 Email Triage OpenEnv Agent

An OpenEnv-based multi-task agent that classifies emails across **support**, **sentiment**, and **priority**, built for the **Meta PyTorch Hackathon x Scaler**.

---

## 🧠 Overview

This project implements a custom OpenEnv environment and an agent (`inference.py`) that interacts with it via HTTP (Hugging Face Space).

The agent performs **3 distinct tasks**:

* 📩 **Support Classification** → `support`
* 😊 **Sentiment Analysis** → `positive`
* ⚡ **Priority Detection** → `high`

Each task is evaluated using custom graders and validated through the OpenEnv pipeline.

---

## ⚙️ Architecture

```text
Agent (inference.py)
        ↓
HTTP Calls (reset / step)
        ↓
OpenEnv Server (HF Space)
        ↓
Environment Logic + Graders
        ↓
Validator (Phase 2)
```

---

## 📁 Project Structure

```text
.
├── inference.py              # Agent logic (LLM + task execution)
├── openenv.yaml             # Task registry (critical for validation)
├── server/
│   ├── __init__.py          # Required for module imports
│   └── environment.py       # Environment + graders
├── models.py                # Action, Observation, State
├── requirements.txt
```

---

## 🧪 Tasks & Graders

Defined in `openenv.yaml`:

```yaml
tasks:
  - id: "support_task"
    grader: "server.environment:grade_support"

  - id: "sentiment_task"
    grader: "server.environment:grade_sentiment"

  - id: "priority_task"
    grader: "server.environment:grade_priority"
```

Each grader returns a score strictly in **(0, 1)** to satisfy validator constraints.

---

## 🤖 Agent Behavior

The agent:

* Calls LLM via OpenAI-compatible API (LiteLLM proxy)
* Uses deterministic logic to ensure task diversity
* Interacts with environment via:

  * `POST /reset`
  * `POST /step`

---

## 📊 Example Output

```text
[START] task=support_task env=openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=support reward=0.80 done=true error=null
[END] success=true steps=1 score=0.80 rewards=0.80

[START] task=sentiment_task env=openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=positive reward=0.60 done=true error=null
[END] success=true steps=1 score=0.60 rewards=0.60

[START] task=priority_task env=openenv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=high reward=0.70 done=true error=null
[END] success=true steps=1 score=0.70 rewards=0.70
```

---

## 🧠 Key Learnings

### 1. STDOUT as a Contract

The validator parses `[START]`, `[STEP]`, `[END]` logs — not just code or YAML.

---

### 2. Task Detection via Logs

Tasks are counted from:

```text
[START] task=<task_id>
```

Not from internal loops.

---

### 3. OpenEnv Runtime Requirement

Direct environment calls fail validation.
Agent must interact via **HTTP / deployed environment**.

---

### 4. Hidden Validator Constraints

* Minimum 3 tasks required
* Rewards must be strictly in **(0,1)**
* Graders must be importable via `openenv.yaml`

---

### 5. Debugging Black-Box Systems

This project required reverse-engineering validator behavior using:

* output inspection
* iterative testing
* system-level reasoning

---

## 🔗 Links

* 🌐 Hugging Face Space:
  [https://huggingface.co/spaces/adityakumarbichhave/email-triage-env](https://huggingface.co/spaces/adityakumarbichhave/email-triage-env)

* 💻 GitHub Repo:
  [https://github.com/adityabichhave/email-triage-openenv](https://github.com/adityabichhave/email-triage-openenv)

---

## 🙌 Acknowledgements

Built as part of the **Meta PyTorch Hackathon x Scaler School of Technology**.

---


