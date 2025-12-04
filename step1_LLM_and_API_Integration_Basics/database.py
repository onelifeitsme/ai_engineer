from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import declarative_base
from config import settings
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession


engine = create_async_engine(url=settings.DATABASE_URL_asyncpg)

async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session


Base = declarative_base()
