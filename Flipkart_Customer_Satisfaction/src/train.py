#!/usr/bin/env python3
"""
src/train.py

Usage:
    python src/train.py --data-path ../data/flipkart.csv --target "CSAT Score" --sample 20000

What it does:
- Loads CSV
- Preprocesses (memory-safe)
- Trains a RandomForest (fast config)
- Saves model, preprocessor metadata, metrics, sample predictions, and feature importances
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


def load_data(path: Path, sample: int = None) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    if sample is not None and sample > 0 and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        print(f"Using random sample of {sample} rows (from {len(df)} rows).")
    print(f"Loaded dataframe shape: {df.shape}")
    return df


def detect_target(df: pd.DataFrame, explicit_target: str = None) -> str:
    if explicit_target and explicit_target in df.columns:
        print(f"Using explicit target column: {explicit_target}")
        return explicit_target
    # heuristics
    for name in df.columns:
        lname = name.lower()
        if any(k in lname for k in ("satisf", "rating", "score", "csat")):
            print(f"Auto-detected target column: {name}")
            return name
    # fallback: last numeric column, else last column
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print(f"No obvious target found: using last numeric column: {num_cols[-1]}")
        return num_cols[-1]
    print(f"No numeric columns found: using last column as target: {df.columns[-1]}")
    return df.columns[-1]


def basic_clean_fill(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows with more than half columns missing
    thresh = df.shape[1] // 2
    before = df.shape[0]
    df = df[df.isnull().sum(axis=1) <= thresh].reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped rows with >50% missing: {before - after} rows removed")
    # fill numeric -> median, categorical -> mode (string)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            df[col] = df[col].fillna(med)
        else:
            mode = df[col].mode()
            fill_val = mode.iloc[0] if not mode.empty else "NA"
            df[col] = df[col].fillna(fill_val)
    return df


def build_preprocessor(X: pd.DataFrame, top_k: int = 30, freq_threshold: int = 100) -> Tuple[pd.DataFrame, Dict]:
    """
    Return:
      - X_enc: transformed numeric DataFrame (dense)
      - preprocessor: dict with metadata (which columns were freq-encoded, which topk, what top values)
    Strategy:
      - numeric columns: keep as-is
      - categorical:
          * if nunique > freq_threshold => frequency encoding (value -> normalized frequency)
          * else => keep top_k categories, replace others with __OTHER__ and one-hot via get_dummies
    """
    print("Building preprocessor ...")
    X = X.copy()
    numeric = X.select_dtypes(include=[np.number]).copy()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    preproc = {"freq_encoded": [], "onehot": {}, "numeric_cols": numeric.columns.tolist()}

    X_enc = numeric

    for col in cat_cols:
        nunique = X[col].nunique(dropna=False)
        if nunique == 0:
            continue
        if nunique > freq_threshold:
            # frequency encoding
            freqs = X[col].value_counts(normalize=True)
            new_col = f"{col}__freq"
            X_enc[new_col] = X[col].map(freqs).fillna(0.0)
            preproc["freq_encoded"].append({"col": col, "new_col": new_col})
            print(f"Column '{col}': freq-encoded (nunique={nunique}) -> {new_col}")
        else:
            # top-k + one-hot
            k = top_k if nunique > top_k else nunique
            top_vals = X[col].value_counts().nlargest(k).index.tolist()
            tmp = X[col].where(X[col].isin(top_vals), other="__OTHER__")
            dummies = pd.get_dummies(tmp, prefix=col, drop_first=True)
            # optionally reduce dummies if too many columns (avoid explosion)
            # no further reduction here, but could limit to top variance columns if needed
            X_enc = pd.concat([X_enc, dummies], axis=1)
            preproc["onehot"][col] = {"top_k": k, "top_vals": top_vals, "dummy_cols": dummies.columns.tolist()}
            print(f"Column '{col}': one-hot top {k} (nunique={nunique}) -> {len(dummies.columns)} columns")

    # final cleanup: replace inf/nan
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"Feature matrix shape after encoding: {X_enc.shape}")
    return X_enc, preproc


def reduce_features_by_variance(X: pd.DataFrame, max_features: int = 2000) -> pd.DataFrame:
    if X.shape[1] <= max_features:
        return X
    variances = X.var().sort_values(ascending=False)
    keep = variances.head(max_features).index.tolist()
    print(f"Reducing features by variance: {X.shape[1]} -> {len(keep)}")
    return X[keep]


def train_model(X: pd.DataFrame, y: pd.Series, is_regression: bool,
                n_estimators: int = 50, max_depth: int = 10) -> object:
    if is_regression:
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
    else:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
    print("Training model ...")
    model.fit(X, y)
    print("Model training finished.")
    return model


def evaluate_and_save(model, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path, X_columns: List[str]):
    print("Evaluating model ...")
    preds = model.predict(X_test)
    metrics = {}
    if pd.api.types.is_numeric_dtype(y_test):
        metrics["mse"] = float(mean_squared_error(y_test, preds))
    else:
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
    # save metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "evaluation.txt").write_text("Metrics:\n" + json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out_dir}/metrics.json")

    # save sample predictions
    sample_df = X_test.copy()
    sample_df["y_true"] = y_test.values
    sample_df["y_pred"] = preds
    sample_out = out_dir.parent.parent / "reports" / "predictions_sample.csv"
    sample_out.parent.mkdir(parents=True, exist_ok=True)
    sample_df.head(200).to_csv(sample_out, index=False)
    print(f"Saved sample predictions to {sample_out}")

    # feature importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        try:
            feat_imp = pd.Series(importances, index=X_columns).sort_values(ascending=False)
        except Exception:
            # fallback if mismatch
            feat_imp = pd.Series(importances).sort_values(ascending=False)
        top = feat_imp.head(50)
        fig_path = out_dir.parent.parent / "model_explainability" / "feature_importances.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 6))
        top.plot.bar()
        plt.title("Top feature importances")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        top.to_csv(str(fig_path.parent / "feature_importances.csv"))
        print(f"Saved feature importances plot to {fig_path}")


def save_artifacts(model, preprocessor: Dict, out_root: Path):
    models_dir = out_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "rf_model.joblib")
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    print(f"Saved model -> {models_dir/'rf_model.joblib'} and preprocessor -> {models_dir/'preprocessor.joblib'}")


def main(args):
    project_root = Path(args.project_root).resolve()
    data_path = Path(args.data_path).resolve()
    out_root = project_root

    os.makedirs(out_root, exist_ok=True)
    # load
    df = load_data(data_path, sample=args.sample)

    # detect target
    target = detect_target(df, explicit_target=args.target)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    # clean/fill
    df_clean = basic_clean_fill(df)

    # split X,y
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    # preprocessor
    X_enc, preproc = build_preprocessor(X, top_k=args.top_k, freq_threshold=args.freq_threshold)

    # optional feature reduction
    X_enc = reduce_features_by_variance(X_enc, max_features=args.max_features)

    # train-test
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=args.test_size, random_state=42)

    is_reg = pd.api.types.is_numeric_dtype(y_train)
    print(f"Training task type: {'regression' if is_reg else 'classification'}")

    # train (small/faster config by default)
    model = train_model(X_train, y_train, is_reg, n_estimators=args.n_estimators, max_depth=args.max_depth)

    # evaluate and save reports
    evaluate_and_save(model, X_test, y_test, out_root / "model_building", X_enc.columns.tolist())

    # save model + preprocessor metadata
    save_artifacts(model, {"preprocessor": preproc, "target": target}, out_root)

    print("All done. Artifacts are in:", out_root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".", help="Project root (where models/, reports/ will be created)")
    p.add_argument("--data-path", default="../data/flipkart.csv", help="Path to csv dataset")
    p.add_argument("--target", default=None, help="Target column name (auto-detected if omitted)")
    p.add_argument("--sample", type=int, default=20000, help="Number of rows to sample (set 0 or None to use full dataset)")
    p.add_argument("--top-k", type=int, default=30, help="Top-k categories to keep for one-hot")
    p.add_argument("--freq-threshold", type=int, default=100, help="If categorical unique values > this -> frequency encode")
    p.add_argument("--max-features", type=int, default=2000, help="Keep top N features by variance after encoding")
    p.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    p.add_argument("--n-estimators", type=int, default=30, help="Number of trees for RandomForest")
    p.add_argument("--max-depth", type=int, default=10, help="Max depth for RandomForest")
    args = p.parse_args()
    main(args)
