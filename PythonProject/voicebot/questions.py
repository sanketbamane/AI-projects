"""Question bank and retrieval for roles."""
from typing import List, Dict
from .config import SAMPLE_QUESTIONS, DEFAULT_ROLES


def get_roles() -> List[str]:
    return list(DEFAULT_ROLES)


def get_questions(role: str) -> List[Dict]:
    """Return list of question dicts for the given role. Falls back to AI Engineer if unknown."""
    if role in SAMPLE_QUESTIONS:
        return SAMPLE_QUESTIONS[role]
    # fallback
    return SAMPLE_QUESTIONS.get('AI Engineer', [])
