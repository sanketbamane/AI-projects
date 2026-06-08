from fastapi import FastAPI
from fastapi import Depends
from recommendation import get_recommendations
from forecast import forecast_inventory

from sqlalchemy.orm import Session

import joblib

import crud
import models
import schemas

from database import engine
from database import SessionLocal

models.Base.metadata.create_all(
    bind=engine
)

app = FastAPI(
    title="AutoResilience AI"
)

model = joblib.load(
    "risk_model.pkl"
)

def get_db():

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()

@app.get("/")
def home():

    return {
        "message":
        "AutoResilience AI Running"
    }

@app.get("/suppliers")
def get_suppliers(
        db:Session = Depends(get_db)
):

    return crud.get_suppliers(db)

@app.post("/suppliers")
def create_supplier(
        supplier:schemas.SupplierCreate,
        db:Session=Depends(get_db)
):

    prediction = model.predict(
    [[
        supplier.lead_time,
        supplier.on_time_delivery,
        supplier.defect_rate,
        supplier.cost_score
    ]]
    )[0]

    risk_score = float(prediction)

    return crud.create_supplier(
        db,
        supplier,
        risk_score
    )

@app.get("/recommend/{supplier_id}")
def recommend_supplier(
        supplier_id:int
):

    return get_recommendations(
        supplier_id
    )

@app.get("/forecast")
def inventory_forecast():

    return forecast_inventory()

@app.post("/predict-risk")
def predict_risk(data:dict):

    prediction = model.predict(
    [[
        data["lead_time"],
        data["on_time_delivery"],
        data["defect_rate"],
        data["cost_score"]
    ]]
    )[0]

    return {
        "risk": int(prediction)
    }