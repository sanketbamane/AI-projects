from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import Float
from sqlalchemy import String

from database import Base

class Supplier(Base):

    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True,index=True)

    supplier_name = Column(String)

    country = Column(String)

    lead_time = Column(Integer)

    on_time_delivery = Column(Float)

    defect_rate = Column(Float)

    cost_score = Column(Float)

    risk_score = Column(Float)