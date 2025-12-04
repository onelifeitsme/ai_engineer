from database import Base
from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column


class Product(Base):
    __tablename__ = "product"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    price: Mapped[int] = mapped_column(Integer)
