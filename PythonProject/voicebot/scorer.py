"""Scoring engine: simple keyword and length-based heuristics."""
from typing import Dict, List
import re
from .config import SCORING


def _normalize_text(text: str) -> str:
    return re.sub(r'[^a-z0-9\s]', ' ', text.lower())


def evaluate(answer: str, question: Dict) -> Dict:
    """Evaluate a single answer against question dict.

    Returns a dict with keys: score, max_points, matched_keywords, feedback
    """
    text = _normalize_text(answer or '')
    words = text.split()
    word_count = len(words)

    keywords = [k.lower() for k in question.get('keywords', [])]
    matched: List[str] = []
    for k in keywords:
        if k in text:
            matched.append(k)

    max_points = int(question.get('max_points', 10))

    # Keyword score: proportion of keywords matched
    kw_score = (len(matched) / max(1, len(keywords))) if keywords else 0

    # Length heuristic: reward reasonable length answers (5-150 words)
    if word_count <= 5:
        length_score = 0.2
    elif word_count < 30:
        length_score = 1.0
    elif word_count < 150:
        length_score = 0.9
    else:
        length_score = 0.7

    # Clarity heuristic: very naive - reward punctuation presence
    clarity_score = 1.0 if any(p in answer for p in '.!?') else 0.8

    w_kw = SCORING.get('keyword_weight', 0.7)
    w_len = SCORING.get('length_weight', 0.2)
    w_cl = SCORING.get('clarity_weight', 0.1)

    total_fraction = (w_kw * kw_score) + (w_len * length_score) + (w_cl * clarity_score)

    raw_score = round(total_fraction * max_points)
    raw_score = max(0, min(raw_score, max_points))

    missing = [k for k in keywords if k not in matched]

    feedback_parts = []
    if matched:
        feedback_parts.append(f"Matched keywords: {', '.join(matched)}.")
    if missing:
        feedback_parts.append(f"Consider mentioning: {', '.join(missing)}.")
    if word_count <= 5:
        feedback_parts.append("Your answer was short; add more detail.")

    feedback = ' '.join(feedback_parts) if feedback_parts else 'Good answer.'

    return {
        'score': raw_score,
        'max_points': max_points,
        'matched_keywords': matched,
        'missing_keywords': missing,
        'feedback': feedback,
        'word_count': word_count,
    }
