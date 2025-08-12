import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

doc = """                This Agreement is made between the parties. The first party agrees to deliver the goods.
The second party will make payment within 30 days. Liability is limited to direct damages only.
Confidentiality must be maintained. Termination requires 30 days notice.
"""

sentences = [s.strip() for s in doc.split('.') if s.strip()]
vec = TfidfVectorizer().fit_transform(sentences)
scores = vec.sum(axis=1).A1
ranked = [s for _,s in sorted(zip(scores, sentences), reverse=True)]
summary = '. '.join(ranked[:2]) + '.'
print('Summary:', summary)
