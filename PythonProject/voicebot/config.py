"""Configuration: roles, questions and scoring defaults."""
from pathlib import Path

DEFAULT_ROLES = ['AI Engineer', 'ML Researcher', 'Data Scientist']

# Simple sample question bank. Each question has id, text, keywords, and max_points.
SAMPLE_QUESTIONS = {
    'AI Engineer': [
        {
            'id': 'ai_1',
            'text': 'Explain the difference between supervised and unsupervised learning.',
            'keywords': ['supervised', 'unsupervised', 'labels', 'clustering', 'classification'],
            'max_points': 10,
        },
        {
            'id': 'ai_2',
            'text': 'How would you approach deploying a machine learning model to production?',
            'keywords': ['container', 'monitoring', 'CI/CD', 'model drift', 'scaling'],
            'max_points': 10,
        },
    ],
    'ML Researcher': [
        {
            'id': 'ml_1',
            'text': 'Describe how you would set up an experiment to compare two models.',
            'keywords': ['hypothesis', 'baseline', 'metrics', 'statistical significance', 'control'],
            'max_points': 10,
        },
    ],
    'Data Scientist': [
        {
            'id': 'ds_1',
            'text': 'How do you handle missing data in a dataset?',
            'keywords': ['imputation', 'drop', 'mean', 'median', 'interpolation', 'model-based'],
            'max_points': 10,
        },
    ],
}

# Logging
DEFAULT_SESSION_FILE = Path.cwd() / 'sessions.json'

# Scoring weights (simple heuristic)
SCORING = {
    'keyword_weight': 0.7,
    'length_weight': 0.2,
    'clarity_weight': 0.1,
}
