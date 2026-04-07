from fastapi import FastAPI
from api.routes import router
import uvicorn

app = FastAPI(
    title="Система автоматического вызова транспорта",
    description="Прототип перехода от прогноза отгрузок к операционным решениям",
    version="1.0.0"
)
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)