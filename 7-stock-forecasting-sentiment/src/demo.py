import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

rng = np.random.RandomState(0)
n = 300
prices = np.cumsum(rng.randn(n))*0.5 + 100
# synthetic 'news' sentiments (-1..1)
sentiments = rng.uniform(-0.5,0.5,size=n)
df = pd.DataFrame({'price':prices, 'sentiment':sentiments})
# create lag features
for lag in range(1,6):
    df[f'lag_{lag}'] = df['price'].shift(lag)
df = df.dropna().reset_index(drop=True)
X = df[[f'lag_{l}' for l in range(1,6)]+['sentiment']].values
y = df['price'].values
split = int(0.8*len(X))
model = Ridge()
model.fit(X[:split], y[:split])
preds = model.predict(X[split:])
print('RMSE:', mean_squared_error(y[split:], preds, squared=False))
