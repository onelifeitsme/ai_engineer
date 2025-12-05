import uvicorn
from fastapi import FastAPI
from api.v1.product import router as products_router
from api.v1.chat import router as chats_router


app = FastAPI()
app.include_router(products_router)
app.include_router(chats_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
