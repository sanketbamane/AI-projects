import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rng = np.random.RandomState(0)
n = 1000
t = np.arange(n)
signal = np.sin(t/50.0) + rng.randn(n)*0.1
df = pd.DataFrame({'sensor': signal})
# create lag features
for lag in range(1,6):
    df[f'lag_{lag}'] = df['sensor'].shift(lag)
df = df.dropna().reset_index(drop=True)
X = df[[f'lag_{l}' for l in range(1,6)]].values
y = df['sensor'].values
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
reg = RandomForestRegressor(n_estimators=50, random_state=1)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print("RMSE:", mean_squared_error(y_test, preds, squared=False))
