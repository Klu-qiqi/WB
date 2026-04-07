from fastapi import FastAPI
import uvicorn

from api.routes import router

app = FastAPI(
    title="DispatchFlow MVP",
    description=(
        "Веб-сервис автоматического вызова транспорта: "
        "forecast service + dispatch decision service + model status."
    ),
    version="1.0.0",
)
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
