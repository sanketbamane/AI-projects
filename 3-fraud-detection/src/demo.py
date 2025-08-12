import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
X = rng.randn(500, 6)
# inject anomalies
X[:10] += rng.randn(10,6)*8
iso = IsolationForest(contamination=0.02, random_state=1)
iso.fit(X)
scores = iso.decision_function(X)
preds = iso.predict(X)  # -1 for anomaly, 1 for normal
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
df['anomaly'] = (preds==-1)
print(df['anomaly'].value_counts())
