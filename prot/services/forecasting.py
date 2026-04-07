from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd

from config import PLANNING_HORIZON_STEPS
from models.forecasting import ForecastModelRegistry


class ForecastingService:
    def __init__(self) -> None:
        self.registry: ForecastModelRegistry | None = None
        self.forecast_runs: list[dict[str, Any]] = []
        self.forecast_predictions: dict[str, pd.DataFrame] = {}

    def _get_registry(self) -> ForecastModelRegistry:
        if self.registry is None:
            print("Initializing forecast model registry...")
            self.registry = ForecastModelRegistry()
            print(
                "Forecast model registry initialized:",
                self.registry.active_model_name,
                self.registry.active_model_version,
                self.registry.status,
            )
        return self.registry

    def generate_forecast(
        self,
        raw_data: pd.DataFrame,
        use_enriched_inference: bool = False,
    ) -> dict[str, Any]:
        registry = self._get_registry()
        raw_data = raw_data.copy()
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
        raw_data = raw_data.sort_values(["timestamp", "route_id"]).reset_index(drop=True)

        pred_df, descriptor = registry.predict_recursive(
            raw_data,
            use_enriched_inference=use_enriched_inference,
        )

        pred_df["confidence_score"] = pred_df["predicted_target_2h"].apply(self._estimate_confidence_score)
        pred_df["confidence"] = pred_df["confidence_score"].apply(self._label_confidence)
        pred_df["model_name"] = descriptor.name
        pred_df["model_version"] = descriptor.version
        pred_df["model_type"] = descriptor.model_type
        pred_df["inference_mode"] = descriptor.mode
        pred_df["is_champion"] = descriptor.is_champion

        run_id = str(uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        run_record = {
            "run_id": run_id,
            "model_version": descriptor.version,
            "model_name": descriptor.name,
            "created_at": created_at,
            "horizon_steps": self._infer_horizon_steps(pred_df),
            "status": descriptor.status,
            "inference_mode": descriptor.mode,
        }
        self.forecast_runs.append(run_record)

        pred_df = pred_df.copy()
        pred_df["run_id"] = run_id
        self.forecast_predictions[run_id] = pred_df

        return {
            "run": run_record,
            "model": asdict(descriptor),
            "predictions": self._serialize_predictions(pred_df),
        }

    def get_forecast_by_run(self, run_id: str) -> pd.DataFrame:
        if run_id not in self.forecast_predictions:
            raise KeyError(f"Forecast run not found: {run_id}")
        return self.forecast_predictions[run_id].copy()

    def get_latest_forecast(self) -> pd.DataFrame:
        if not self.forecast_runs:
            raise KeyError("No forecast runs found.")
        return self.get_forecast_by_run(self.forecast_runs[-1]["run_id"])

    def get_forecast_for_warehouse(self, office_from_id: int, run_id: str | None = None) -> dict[str, Any]:
        forecast_df = self.get_forecast_by_run(run_id) if run_id else self.get_latest_forecast()
        warehouse_df = forecast_df[forecast_df["office_from_id"] == office_from_id].copy()
        if warehouse_df.empty:
            raise KeyError(f"No forecast rows for office_from_id={office_from_id}")

        return {
            "office_from_id": office_from_id,
            "run_id": warehouse_df["run_id"].iloc[0],
            "model_name": warehouse_df["model_name"].iloc[0],
            "model_version": warehouse_df["model_version"].iloc[0],
            "forecast": self._serialize_predictions(warehouse_df),
        }

    def get_model_status(self) -> dict[str, Any]:
        registry = self._get_registry()
        return registry.get_registry_status()

    def _serialize_predictions(self, pred_df: pd.DataFrame) -> list[dict[str, Any]]:
        serialized = pred_df.copy()
        serialized["timestamp"] = serialized["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return serialized.to_dict(orient="records")

    def _infer_horizon_steps(self, pred_df: pd.DataFrame) -> int:
        if pred_df.empty:
            return PLANNING_HORIZON_STEPS
        return min(pred_df["timestamp"].nunique(), PLANNING_HORIZON_STEPS)

    def _estimate_confidence_score(self, prediction: float) -> float:
        if prediction <= 5:
            return 0.9
        if prediction <= 15:
            return 0.75
        if prediction <= 30:
            return 0.6
        return 0.5

    def _label_confidence(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"
