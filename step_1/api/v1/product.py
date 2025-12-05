from typing import List, Optional
from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from schemas import product as schemas
from fastapi import Depends, Query
from database import get_async_session
from crud.product import get_products


router = APIRouter()


@router.get("/products", response_model=List[schemas.ProductBase])
async def get_all_products(
    session: AsyncSession = Depends(get_async_session),
    skip: int = Query(0, ge=0),
    limit: Optional[int] = Query(None, ge=1)
):
    products = await get_products(db=session, skip=skip, limit=limit)
    return products
