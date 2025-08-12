import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# synthetic data
rng = np.random.RandomState(42)
n = 1000
X = rng.randn(n, 8)
y = (X[:,0] + X[:,1]*0.5 + rng.randn(n)*0.3 > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf = RandomForestClassifier(n_estimators=50, random_state=1)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("Precision:", precision_score(y_test, preds))
