from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.product import Product
from schemas.product import ProductCreate


async def create_product(db: AsyncSession, product: ProductCreate) -> Product:
    db_product = Product(name=product.name)
    db.add(db_product)
    await db.commit()
    await db.refresh(db_product)
    return db_product


async def get_product(db: AsyncSession, product_id: int) -> Optional[Product]:
    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )
    return result.scalar_one_or_none()


async def get_products(
    db: AsyncSession,
    skip: int = 0,
    limit: int | None = None
) -> List[Product]:
    query = select(Product).offset(skip)

    if limit is not None:
        query = query.limit(limit)

    result = await db.execute(query)
    return result.scalars().all()
