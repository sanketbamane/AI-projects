# Mini AI Interview Screener (Backend Only)

A lightweight backend service built with **FastAPI (Python)** that evaluates candidate answers using either:
- An **LLM** (OpenAI / Anthropic / others), or
- A built‑in **heuristic offline evaluator**.

The service provides two main APIs:
1. **POST /evaluate-answer** – Score a single answer
2. **POST /rank-candidates** – Rank multiple candidates by score

---

## 🚀 Features
- Optional LLM integration (OpenAI supported by default)
- Deterministic offline scoring when no LLM keys are provided
- Async concurrent evaluation for ranking many candidates
- Clean response contracts using Pydantic
- One-file backend for easy review and deployment

---

## 📁 Project Structure
```
mini_ai_interviewer.py   → Main FastAPI application
README.md                → Project documentation
```

---

## 🛠️ Tech Stack
- **Python 3.9+**
- **FastAPI** for API routing
- **Uvicorn** for the ASGI server
- **Pydantic** for request/response modeling
- **OpenAI SDK** (optional – only required if using LLM mode)

---

## 📦 Installation

### 1. Clone the repo
```
git clone <your-repo-url>
cd <your-folder>
```

### 2. Create virtual environment
```
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

If you don't want LLM support, you may omit `openai`.

---

## ⚙️ Environment Variables
The service can run in two modes.

### **Default (Offline heuristic mode)**
No env vars needed.

### **LLM Mode (OpenAI)**
Set:
```
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
export OPENAI_MODEL=gpt-4o-mini   # or any model name
```
Windows PowerShell:
```
setx LLM_PROVIDER "openai"
setx OPENAI_API_KEY "your_key_here"
setx OPENAI_MODEL "gpt-4o-mini"
```

When `LLM_PROVIDER=openai` is set, the backend will attempt to call OpenAI; otherwise it falls back to heuristic.

---

## ▶️ Run the Server
```
uvicorn mini_ai_interviewer:app --reload --port 8000
```
Your API is now available at:
```
http://localhost:8000
```
Interactive docs:
```
http://localhost:8000/docs
```

---

## 🧪 Test the APIs

### 1️⃣ **Evaluate one answer**
```
curl -X POST http://localhost:8000/evaluate-answer \
-H "Content-Type: application/json" \
-d '{"text": "Candidate says: I would use indexing and caching to reduce latency."}'
```
Response example:
```json
{
  "score": 4,
  "summary": "The candidate suggests indexing and caching to reduce latency",
  "improvement": "Add a rationale or trade-off for the design choice"
}
```

---

### 2️⃣ **Rank multiple candidates**
```
curl -X POST http://localhost:8000/rank-candidates \
-H "Content-Type: application/json" \
-d '{"answers": [
  "Candidate says: I would start by designing a scalable architecture...",
  "Candidate says: Use a list maybe.",
  "Candidate says: I would use concurrency and caching for performance."
]}'
```

Response example:
```json
[
  {"text": "Candidate says: ...", "score": 5, ...},
  {"text": "Candidate says: ...", "score": 4, ...},
  {"text": "Candidate says: ...", "score": 2, ...}
]
```

---

## 🧠 How It Works
### ✔ If LLM is enabled:
- Sends answer to OpenAI with a strict JSON prompt
- Tries to parse `{score, summary, improvement}`
- Falls back to heuristic if JSON is malformed

### ✔ If LLM is NOT enabled:
- Uses length, keywords, and reasoning indicators
- Generates summary + improvement suggestions
- Guarantees deterministic scoring

---

## 🧭 Why This Architecture?
- **FastAPI** gives clean, typed endpoints and async processing
- **Heuristic fallback** ensures the service always works offline
- **Optional LLM** allows powerful evaluation when available
- **Single-file project** keeps the assignment simple and easy to review

---

## 📌 Future Enhancements
- Add Dockerfile
- Add pytest unit tests
- Support Claude / LLaMA / Groq models
- Store candidate history in SQLite/Postgres
- Add authentication

---

## 📞 Support
If you need help extending or deploying this project, feel free to reach out.

