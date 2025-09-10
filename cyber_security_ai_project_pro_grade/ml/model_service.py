from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import os, json
app = FastAPI(title='Model Service')

class Req(BaseModel):
    message: str
    source: str

CLASSIFIER_PATH = os.getenv('MODEL_PATH', 'ml/model.joblib')
model = None
try:
    model = load(CLASSIFIER_PATH)
except Exception as e:
    print('Model load failed:', e)

@app.post('/predict')
def predict(r: Req):
    if model is None:
        return {'score': 0.0, 'label': 0}
    text = f"[{r.source}] " + r.message
    prob = float(model.predict_proba([text])[0,1])
    label = int(prob >= 0.5)
    return {'score': prob, 'label': label}

@app.post('/explain')
def explain(r: Req):
    if model is None:
        return {'shap': [], 'note': 'no-model'}
    # Fast surrogate explanation: map RandomForest feature_importances_ to TF-IDF feature names
    try:
        tfidf = model.named_steps['tfidf']
        rf = model.named_steps['rf']
        feat_names = tfidf.get_feature_names_out()
        importances = rf.feature_importances_
        # Some models may have fewer importance entries if n_features different; clip
        n = min(len(feat_names), len(importances))
        pairs = list(zip(feat_names[:n], importances[:n]))
        # sort by importance desc and pick top 10
        top = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
        return {'shap': [{'feature': f, 'importance': float(imp)} for f,imp in top], 'note': 'surrogate-feature-importances'}
    except Exception as e:
        return {'shap': [], 'note': f'error:{e}'}
