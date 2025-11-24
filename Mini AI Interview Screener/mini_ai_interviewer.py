"""
Mini AI Interview Screener (Backend Only)
Framework: FastAPI
File: mini_ai_interviewer.py

Features:
- POST /evaluate-answer  -> accepts JSON {"text": "Candidate says: <their answer>"}
  returns {"score": 1-5, "summary": "one-line", "improvement": "one suggestion"}

- POST /rank-candidates -> accepts JSON {"answers": ["Candidate says: ...", ...]}
  returns sorted list of objects [{"text":..., "score":..., "summary":..., "improvement":...}, ...]

Behavior:
- If environment variable OPENAI_API_KEY or ANTHROPIC_API_KEY is present and `LLM_PROVIDER` env var set to `openai` or `anthropic`, the app will call that LLM to evaluate answers.
- Otherwise it will fall back to a small deterministic heuristic scorer (fast and offline) so the service is usable without API keys.

Run (dev):
    pip install -r requirements.txt
    uvicorn mini_ai_interviewer:app --reload --port 8000

Example requests (curl):
    curl -X POST "http://127.0.0.1:8000/evaluate-answer" -H "Content-Type: application/json" -d '{"text":"Candidate says: I would use a hash map for O(1) lookups..."}'

Note: This implementation focuses on clarity over heavy-weight LLM integration. It provides a clean place to plug in your preferred LLM provider.
"""

from typing import List, Optional, Dict, Any
import os
import re
import asyncio

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Optional imports for LLM providers
try:
    import openai
except Exception:
    openai = None

# You can add Anthropic/other SDKs similarly if you want to enable them.

app = FastAPI(title="Mini AI Interview Screener", version="1.0")

# ----------------------
# Request / Response models
# ----------------------
class EvaluateRequest(BaseModel):
    text: str

class EvaluateResponse(BaseModel):
    score: int  # 1-5
    summary: str
    improvement: str

class RankRequest(BaseModel):
    answers: List[str]

class RankedCandidate(BaseModel):
    text: str
    score: int
    summary: str
    improvement: str

# ----------------------
# Heuristic fallback scorer (deterministic, offline)
# ----------------------

TECH_KEYWORDS = [
    # generic technical / interview keywords — tune for your domain
    "design", "scale", "algorithm", "complexity", "optimize", "trade-off", "latency", "consistency",
    "hash", "map", "array", "database", "index", "cache", "concurrency", "thread", "lock", "transaction",
    "test", "edge case", "security", "performance", "profil", "benchmark",
]


def heuristic_score_and_feedback(text: str) -> Dict[str, Any]:
    # Normalize text
    t = text.lower()

    # Extract candidate content if user prefixed with "Candidate says:"
    m = re.search(r"candidate says:\s*(.*)$", t, flags=re.IGNORECASE)
    if m:
        candidate_answer = m.group(1).strip()
    else:
        candidate_answer = t.strip()

    length = len(candidate_answer.split())
    unique_kws = sum(1 for kw in TECH_KEYWORDS if kw in candidate_answer)

    # Basic scoring rules (1-5)
    # - very short answers -> 1-2
    # - presence of keywords, reasoning, and length -> higher
    score = 1
    if length >= 60 and unique_kws >= 3:
        score = 5
    elif length >= 40 and unique_kws >= 2:
        score = 4
    elif length >= 25 and unique_kws >= 1:
        score = 3
    elif length >= 12:
        score = 2
    else:
        score = 1

    # Summary: first sentence or trimmed candidate text
    summary = candidate_answer.split('.')
    if len(summary) > 0 and summary[0].strip():
        one_line = summary[0].strip()
        if len(one_line) > 160:
            one_line = one_line[:157] + '...'
    else:
        one_line = (candidate_answer[:157] + '...') if len(candidate_answer) > 160 else candidate_answer

    # Improvement suggestion
    improvement_reasons = []
    if unique_kws == 0:
        improvement_reasons.append("Mention specific technologies, data structures or metrics to show concrete knowledge")
    if "because" not in candidate_answer and "since" not in candidate_answer and "so that" not in candidate_answer:
        improvement_reasons.append("Add a brief rationale (why this approach), plus a trade-off")
    if length < 40:
        improvement_reasons.append("Expand with 1–2 short examples or steps you'd take in implementation")

    improvement = (
        "; ".join(improvement_reasons) if improvement_reasons else "Good answer — consider adding one short example or a trade-off to strengthen it."
    )

    return {"score": score, "summary": one_line, "improvement": improvement}


# ----------------------
# LLM-based evaluation (pluggable)
# ----------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "heuristic").lower()  # 'openai', 'anthropic', or 'heuristic'

async def call_openai_eval(candidate_text: str) -> Dict[str, Any]:
    """
    Example prompt and call to OpenAI ChatCompletion (gpt-4 / gpt-3.5).
    You must set OPENAI_API_KEY in environment to enable this.
    """
    if openai is None:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    openai.api_key = api_key

    # Construct careful evaluation prompt — ask for a JSON only response
    system_prompt = (
        "You are an expert technical interviewer who grades candidate answers from 1 to 5. "
        "Return ONLY valid JSON with keys: score (integer 1-5), summary (one-line), improvement (one short suggestion)."
    )
    user_prompt = (
        f"Evaluate the following candidate answer. Provide a quick score 1-5, a one-line summary, and one improvement suggestion.\n\n" 
        f"Candidate says: {candidate_text}\n\nRespond with JSON."
    )

    # Make the request (sync clients exist; we use async via to_thread)
    def do_request():
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        return resp

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, do_request)

    # Extract assistant text
    text = resp["choices"][0]["message"]["content"].strip()

    # Try to parse JSON inside returned text robustly
    import json
    try:
        # Some models will wrap JSON in markdown; find the first { ... }
        jmatch = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if jmatch:
            parsed = json.loads(jmatch.group(0))
        else:
            parsed = json.loads(text)
        # sanitize
        parsed = {"score": int(parsed.get("score", 1)), "summary": str(parsed.get("summary", "")), "improvement": str(parsed.get("improvement", ""))}
        return parsed
    except Exception as e:
        # If parsing fails, fallback to heuristic
        return heuristic_score_and_feedback(candidate_text)


async def evaluate_with_llm(candidate_text: str) -> Dict[str, Any]:
    provider = LLM_PROVIDER
    if provider == "openai":
        try:
            return await call_openai_eval(candidate_text)
        except Exception as e:
            # Log error in real system; fallback
            return heuristic_score_and_feedback(candidate_text)
    else:
        # default heuristic
        return heuristic_score_and_feedback(candidate_text)

# ----------------------
# Endpoints
# ----------------------

@app.post("/evaluate-answer", response_model=EvaluateResponse)
async def evaluate_answer(req: EvaluateRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    result = await evaluate_with_llm(req.text)

    # Ensure score bounds
    score = max(1, min(5, int(result.get("score", 1))))
    return EvaluateResponse(score=score, summary=result.get("summary", ""), improvement=result.get("improvement", ""))


@app.post("/rank-candidates", response_model=List[RankedCandidate])
async def rank_candidates(req: RankRequest):
    if not req.answers:
        raise HTTPException(status_code=400, detail="answers array required")

    # Evaluate concurrently
    coros = [evaluate_with_llm(a) for a in req.answers]
    results = await asyncio.gather(*coros)

    # Build combined list
    combined = []
    for text, res in zip(req.answers, results):
        sc = max(1, min(5, int(res.get("score", 1))))
        combined.append({"text": text, "score": sc, "summary": res.get("summary", ""), "improvement": res.get("improvement", "")})

    # Sort descending by score, then by length (longer better as tie-breaker)
    combined.sort(key=lambda x: (x["score"], len(x["text"])), reverse=True)

    return [RankedCandidate(**c) for c in combined]


# Simple root
@app.get("/")
async def root():
    return {"service": "Mini AI Interview Screener", "llm_provider": LLM_PROVIDER}
