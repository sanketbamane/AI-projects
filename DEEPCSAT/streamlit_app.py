# streamlit_app.py
import os
import streamlit as st
import numpy as np

# Adjust this to your file name
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "model.joblib")

# Optional: if you plan to download from S3 locally, set these env vars and use boto3 logic
# MODEL_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET")
# MODEL_S3_KEY = os.environ.get("MODEL_S3_KEY")

@st.cache_resource  # keeps model loaded across reruns
def load_model(path=LOCAL_MODEL_PATH):
    # Try sklearn joblib
    try:
        import joblib
        model = joblib.load(path)
        st.session_state["_model_type"] = "sklearn"
        return model
    except Exception as e_joblib:
        # Try TensorFlow/Keras
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(path)
            st.session_state["_model_type"] = "tensorflow"
            return model
        except Exception as e_tf:
            # Try PyTorch
            try:
                import torch
                model = torch.load(path, map_location="cpu")
                st.session_state["_model_type"] = "pytorch"
                return model
            except Exception as e_torch:
                raise RuntimeError(
                    f"Failed to load model. joblib error: {e_joblib}; tf error: {e_tf}; torch error: {e_torch}"
                )

model = load_model()

st.title("DeepCSAT — Local model test")

st.write("Model loaded. Type example input (comma-separated) or upload CSV for batch.")

# SINGLE INPUT
with st.form("single"):
    raw = st.text_area("Single-sample (comma-separated features)", value="0.1,0.2,0.3")
    submit = st.form_submit_button("Predict single")
if submit:
    try:
        arr = np.array([float(x.strip()) for x in raw.split(",")])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Shape check for sklearn if available
        if st.session_state.get("_model_type") == "sklearn" and hasattr(model, "n_features_in_"):
            expected = model.n_features_in_
            if arr.shape[1] != expected:
                st.error(f"Feature mismatch: model expects {expected} features but got {arr.shape[1]}.")
            else:
                preds = model.predict(arr)
                st.success(f"Prediction: {preds.tolist()}")
        else:
            preds = model.predict(arr)
            st.success(f"Prediction: {preds.tolist()}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# BATCH (CSV)
uploaded = st.file_uploader("Upload CSV (each row = features, no header)", type=["csv"])
if uploaded is not None:
    import pandas as pd
    df = pd.read_csv(uploaded, header=None)
    st.write("Data sample:")
    st.dataframe(df.head())
    if st.button("Run batch prediction"):
        try:
            X = df.values
            if st.session_state.get("_model_type") == "sklearn" and hasattr(model, "n_features_in_"):
                if X.shape[1] != model.n_features_in_:
                    st.error(f"Feature mismatch: model expects {model.n_features_in_}, dataset has {X.shape[1]}.")
                else:
                    preds = model.predict(X)
            else:
                preds = model.predict(X)
            df["prediction"] = preds
            st.download_button("Download predictions CSV", df.to_csv(index=False), "preds.csv")
            st.success("Batch predictions done.")
        except Exception as e:
            st.error(f"Error during batch predict: {e}")
