from lib2to3.pytree import Base
from pydantic import BaseModel

# Request Template
class Data(BaseModel):
    a: float
    b: float
    c: float
    d: float
