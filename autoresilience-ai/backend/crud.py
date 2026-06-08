from sqlalchemy.orm import Session

import models

def create_supplier(
        db:Session,
        supplier,
        risk_score
):

    db_supplier = models.Supplier(

        supplier_name=supplier.supplier_name,
        country=supplier.country,
        lead_time=supplier.lead_time,
        on_time_delivery=supplier.on_time_delivery,
        defect_rate=supplier.defect_rate,
        cost_score=supplier.cost_score,
        risk_score=risk_score
    )

    db.add(db_supplier)

    db.commit()

    db.refresh(db_supplier)

    return db_supplier

def get_suppliers(db:Session):

    return db.query(
        models.Supplier
    ).all()