from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.dispatcher import DispatchService
from services.forecasting import ForecastingService

router = APIRouter()
forecast_svc = ForecastingService()
dispatch_svc = DispatchService()


class ConfirmRequest(BaseModel):
    recommendation_id: str
    user_id: str | None = None


class OverrideRequest(BaseModel):
    recommendation_id: str
    new_vehicles_needed: int = Field(ge=0)
    comment: str
    user_id: str | None = None


def _validate_input_frame(df: pd.DataFrame) -> None:
    required = {"route_id", "office_from_id", "timestamp"}
    if not required.issubset(df.columns):
        raise HTTPException(400, f"Отсутствуют обязательные колонки: {sorted(required - set(df.columns))}")


@router.post("/forecast/run")
async def run_forecast(
    data: list[dict[str, Any]],
    use_enriched_inference: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        df = pd.DataFrame(data)
        _validate_input_frame(df)
        return forecast_svc.generate_forecast(df, use_enriched_inference=use_enriched_inference)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("/forecast/warehouse/{office_from_id}")
async def get_forecast_for_warehouse(office_from_id: int, run_id: str | None = None) -> dict[str, Any]:
    try:
        return forecast_svc.get_forecast_for_warehouse(office_from_id, run_id=run_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("/dispatch/recommendations")
async def get_dispatch_recommendations(run_id: str | None = None) -> dict[str, Any]:
    return dispatch_svc.get_recommendations(run_id=run_id)


@router.post("/dispatch/recommendations")
async def create_dispatch_recommendations(run_id: str | None = None) -> dict[str, Any]:
    try:
        forecast_df = forecast_svc.get_forecast_by_run(run_id) if run_id else forecast_svc.get_latest_forecast()
        return dispatch_svc.calculate_dispatch_plan(forecast_df, run_id=forecast_df["run_id"].iloc[0])
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/dispatch/confirm")
async def confirm_dispatch(payload: ConfirmRequest) -> dict[str, Any]:
    try:
        return dispatch_svc.confirm_recommendation(payload.recommendation_id, user_id=payload.user_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/dispatch/override")
async def override_dispatch(payload: OverrideRequest) -> dict[str, Any]:
    try:
        return dispatch_svc.override_recommendation(
            recommendation_id=payload.recommendation_id,
            new_vehicles_needed=payload.new_vehicles_needed,
            comment=payload.comment,
            user_id=payload.user_id,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/models/status")
async def get_models_status() -> dict[str, Any]:
    return forecast_svc.get_model_status()


@router.post("/pipeline")
async def full_pipeline(
    raw_data: list[dict[str, Any]],
    use_enriched_inference: bool = Query(default=False),
) -> dict[str, Any]:
    forecast_result = await run_forecast(raw_data, use_enriched_inference=use_enriched_inference)
    run_id = forecast_result["run"]["run_id"]
    dispatch_result = await create_dispatch_recommendations(run_id=run_id)
    return {"forecast": forecast_result, "dispatch_plan": dispatch_result}


@router.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "operational", "service": "dispatchflow-mvp"}
