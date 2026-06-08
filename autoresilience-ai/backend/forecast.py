import pandas as pd

from prophet import Prophet

df = pd.read_csv(
    "inventory.csv"
)

model = Prophet()

model.fit(df)

def forecast_inventory():

    future = model.make_future_dataframe(
        periods=6,
        freq="M"
    )

    result = model.predict(
        future
    )

    return result[
        ["ds","yhat"]
    ].tail(6).to_dict(
        orient="records"
    )