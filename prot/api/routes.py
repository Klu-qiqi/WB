from fastapi import APIRouter, HTTPException
import pandas as pd
from services.forecasting import ForecastingService
from services.dispatcher import DispatchService

router = APIRouter()
forecast_svc = ForecastingService()
dispatch_svc = DispatchService()

@router.post("/predict")
async def predict(data: list[dict]):
    """Прием статусов и возврат прогноза по маршрутам"""
    try:
        df = pd.DataFrame(data)
        required = {"route_id", "office_from_id", "timestamp", "status_1", "status_2", "status_3", "status_4", "status_5", "status_6", "status_7", "status_8"}
        if not required.issubset(df.columns):
            raise HTTPException(400, f"Отсутствуют колонки: {required - set(df.columns)}")
        return forecast_svc.generate_forecast(df).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/dispatch-plan")
async def create_dispatch_plan(forecasts: list[dict]):
    """Преобразование прогноза в заявки на транспорт"""
    df = pd.DataFrame(forecasts)
    if "predicted_target_2h" not in df.columns:
        raise HTTPException(400, "Необходимо поле predicted_target_2h")
    return dispatch_svc.calculate_dispatch_plan(df)

@router.post("/pipeline")
async def full_pipeline(raw_data: list[dict]):
    """Сквозной процесс: данные -> прогноз -> план вызова ТС"""
    forecasts = await predict(raw_data)
    plan = await create_dispatch_plan(forecasts)
    return {"forecast": forecasts, "dispatch_plan": plan}

@router.get("/health")
async def health():
    return {"status": "operational", "service": "auto-dispatch-system"}