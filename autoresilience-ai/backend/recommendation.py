import pandas as pd

from sklearn.neighbors import NearestNeighbors

suppliers = pd.read_csv(
    "suppliers_master.csv"
)

X = suppliers[
[
"lead_time",
"on_time_delivery",
"cost_score"
]
]

model = NearestNeighbors(
    n_neighbors=3
)

model.fit(X)

def get_recommendations(
        supplier_id
):

    current = suppliers[
        suppliers.id == supplier_id
    ]

    idx = current.index[0]

    distances, indices = model.kneighbors(
        [X.iloc[idx]]
    )

    return suppliers.iloc[
        indices[0]
    ].to_dict(
        orient="records"
    )