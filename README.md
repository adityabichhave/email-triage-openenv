# 📧 Email Triage OpenEnv

## 🚀 Overview

This project implements an **OpenEnv-compatible environment** for classifying customer emails into predefined categories. It is designed for reinforcement learning workflows and integrates seamlessly with the OpenEnv ecosystem.

---

## 🎯 Problem Statement

Automatically classify incoming customer emails into one of the following categories:

* **support**
* **sales**
* **complaint**

---

## 🧠 Environment Design

The environment follows the **OpenAI Gym-style interface**:

### 🔹 Observation

```json
{
  "email": "Customer email text"
}
```

### 🔹 Action

```text
support | sales | complaint
```

### 🔹 Reward

* `+1` → Correct classification
* `0` → Incorrect classification

### 🔹 Done

* Always `false` (single-step environment)

---

## 📁 Project Structure

```
email-triage-openenv/
│
├── env.py              # Core environment logic
├── inference.py        # Evaluation script (used in validation)
├── openenv.yaml        # OpenEnv configuration
├── pyproject.toml      # Project metadata (multi-mode support)
├── requirements.txt    # Dependencies
├── Dockerfile          # Container setup
│
├── server/
│   └── app.py          # Flask API server
│
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run the server locally

```bash
python server/app.py
```

Server will start at:

```
http://localhost:7860
```

---

## 🔌 API Endpoints

### 🔹 Reset

```bash
POST /reset
```

**Response:**

```json
{
  "observation": {
    "email": "..."
  },
  "reward": { "value": 0.0 },
  "done": false,
  "info": {}
}
```

---

### 🔹 Step

```bash
POST /step
```

**Body:**

```json
{
  "action": "support"
}
```

---

### 🔹 State

```bash
GET /state
```

---

### 🔹 Health Check

```bash
GET /
```

Returns:

```
OK
```

---

## 🧪 Inference Script

Run locally:

```bash
python inference.py
```

This will:

* Reset environment
* Execute sample actions
* Output average score

---

## 🐳 Deployment

This project is containerized using Docker and deployed on **Hugging Face Spaces**.

* Exposes port `7860`
* Uses Flask server
* Fully compatible with OpenEnv validation pipeline

---

## ✅ Features

* OpenEnv compliant
* REST API interface
* Lightweight and fast
* Easy to extend for RL agents
* Multi-mode deployment ready

---

## 👤 Author

**Aditya Kumar Bichhave**

---

## 📌 Notes

* Ensure `env.py` is in root for compatibility with validation
* Server runs from `server/app.py`
* Inference script must execute without errors

---
