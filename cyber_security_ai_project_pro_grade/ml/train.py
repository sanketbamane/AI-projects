import argparse, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump
from datetime import datetime
import mlflow, mlflow.sklearn
from api.config import settings

def load_data(path):
    df = pd.read_csv(path)
    df['text'] = '[' + df['source'].astype(str) + '] ' + df['message'].astype(str)
    return df

def main(args):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment(args.experiment or 'threat_detection')
    with mlflow.start_run():
        df = load_data(args.input)
        X = df['text']; y = df['label'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=20000)), ('rf', RandomForestClassifier(n_estimators=args.trees, n_jobs=-1, random_state=42))])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:,1]
        print(classification_report(y_test, preds))
        try:
            auc = roc_auc_score(y_test, probs)
            print('AUC:', auc)
            mlflow.log_metric('auc', float(auc))
        except Exception:
            pass
        os.makedirs('ml', exist_ok=True)
        out = args.output or 'ml/model.joblib'
        dump(pipeline, out)
        mlflow.log_artifact(out, artifact_path='models')
        mlflow.sklearn.log_model(pipeline, 'sklearn-model')
        print('Saved model to', out)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/logs.csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--experiment', default=None)
    args = parser.parse_args()
    main(args)
