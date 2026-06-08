from pydantic import BaseModel

class SupplierCreate(BaseModel):

    supplier_name:str
    country:str

    lead_time:int
    on_time_delivery:float
    defect_rate:float
    cost_score:float

class SupplierResponse(SupplierCreate):

    id:int
    risk_score:float

    class Config:
        from_attributes=True