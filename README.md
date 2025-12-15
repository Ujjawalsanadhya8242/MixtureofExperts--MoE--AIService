# Mixture of Experts (MoE) AI Service

A **production-oriented Mixture of Experts (MoE) system** that dynamically routes user queries to the most relevant domain-specific LLM using a **learned gating model**, exposed via a **FastAPI inference service**.

This is not prompt-routing with `if/else`. The routing decision is **learned**, **data-driven**, and **extensible**.

---

## 🚀 What This Project Does

* Accepts a natural language query via an API
* Uses a trained ML **gating classifier** to select the best expert
* Routes the query to a **domain-specific LLM**
* Returns the expert name + generated response

Currently supported domains:

* 🏥 Healthcare
* 💰 Finance
* 🔐 Cybersecurity

---

## 🧠 Architecture Overview

```
Client
  ↓
FastAPI Service (/predict)
  ↓
Gating Model (TF-IDF + Naive Bayes)
  ↓
Selected Expert LLM
  ↓
Generated Response
```

The system cleanly separates **routing logic** from **generation logic**, which is the core idea behind MoE systems.

---

## 🧩 Core Components

### 1. Gating Network (Learned Router)

* Model: `Multinomial Naive Bayes`
* Features: `TF-IDF (unigrams + bigrams)`
* Input: user query text
* Output: expert label (`healthcare | finance | cybersecurity`)

Training logic lives in:

```
gating/train_gating.py
```

Models are serialized using `joblib`.

---

### 2. Domain Experts (LLMs)

Each expert is a **specialized LLM**, independently swappable:

| Domain        | Model                               |
| ------------- | ----------------------------------- |
| Healthcare    | `llama-2-7b-HealthCareMagic`        |
| Finance       | `Finance-Llama-8B`                  |
| Cybersecurity | `Llama-3-8B-Instruct-Cybersecurity` |

Experts expose a unified interface:

```python
predict(text: str) -> str
```

---

### 3. Inference API (FastAPI)

```http
POST /predict
```

**Request**:

```json
{
  "query": "Is this ransomware attack dangerous?"
}
```

**Response**:

```json
{
  "expert": "CybersecurityExpert",
  "response": "..."
}
```

---

## 📁 Project Structure

```
MOE/
├── configs/
│   ├── gating_model.joblib
│   └── vectorizer.joblib
├── experts/
│   ├── healthcare.py
│   ├── finance.py
│   └── cybersecurity.py
├── gating/
│   ├── gate.py
│   └── train_gating.py
├── services/
│   └── moe_service.py
├── dataset.csv
├── requirements.txt
```

---

## ▶️ Running the Service

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train gating model (optional)

```bash
python gating/train_gating.py
```

### 3️⃣ Start API server

```bash
uvicorn services.moe_service:app --reload
```

API docs:

```
http://127.0.0.1:8000/docs
```

---

## ❗ Why This Project Matters

Most "MoE" demos:

* Hardcode routing
* Use keyword matching
* Collapse all logic into prompts

This project:

* Uses a **trained classifier** for routing
* Cleanly decouples experts
* Matches real MoE design patterns used in production systems

This is closer to **how scalable AI systems are actually built**.

---

## ⚠️ Limitations (Intentional)

* Single-node inference
* No batching or async generation
* No load balancing across experts

These trade-offs keep the focus on **routing architecture**, not infra noise.

---

## 🔮 Future Improvements

* Transformer-based gating model
* Confidence-based expert fallback
* Multi-expert aggregation
* GPU-aware expert scheduling
* Streaming responses

---

## 👤 Author
**Ujjawal Sanadhya**

If you understand why learned routing beats prompt routing, you understand this project.

