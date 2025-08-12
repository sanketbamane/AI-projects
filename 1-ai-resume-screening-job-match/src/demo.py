import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resumes = [
    "Software engineer with experience in Python, machine learning, and web development",
    "Data scientist experienced in pandas, scikit-learn, and statistical modeling",
    "Frontend developer skilled in React, TypeScript, and CSS"
]

job_desc = "Looking for a Python machine learning engineer experienced with scikit-learn and data pipelines."

vec = TfidfVectorizer().fit_transform(resumes + [job_desc])
sims = cosine_similarity(vec[-1], vec[:-1]).flatten()
df = pd.DataFrame({'resume': resumes, 'score': sims})
print(df.sort_values('score', ascending=False).to_string(index=False))
