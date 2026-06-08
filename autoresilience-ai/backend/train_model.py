import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import joblib

df = pd.read_csv("suppliers.csv")

X = df[
[
"lead_time",
"on_time_delivery",
"defect_rate",
"cost_score"
]
]

y = df["risk"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X,y)

joblib.dump(
    model,
    "risk_model.pkl"
)

print("Model trained")