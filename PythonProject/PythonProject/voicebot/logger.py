"""Session logging utilities: append session records to a JSON file."""
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
from .config import DEFAULT_SESSION_FILE


def _ensure_file(path: Path):
    if not path.exists():
        path.write_text('[]', encoding='utf-8')


def save_session(record: Dict, path: Path = DEFAULT_SESSION_FILE) -> None:
    path = Path(path)
    _ensure_file(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    data.append(record)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def make_session_record(role: str, questions: List[Dict], results: List[Dict]) -> Dict:
    total_score = sum(r.get('score', 0) for r in results)
    total_max = sum(r.get('max_points', 0) for r in results)
    start_ts = datetime.utcnow().isoformat() + 'Z'
    session = {
        'session_id': start_ts,
        'role': role,
        'start_timestamp': start_ts,
        'end_timestamp': None,
        'questions': [],
        'total_score': total_score,
        'total_max': total_max,
    }
    for q, r in zip(questions, results):
        session['questions'].append({
            'id': q.get('id'),
            'text': q.get('text'),
            'answer': r.get('answer'),
            'score': r.get('score'),
            'max_points': r.get('max_points'),
            'matched_keywords': r.get('matched_keywords', []),
            'feedback': r.get('feedback', ''),
        })
    return session
