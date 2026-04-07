from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from config import (
    AUTO_APPROVE_MAX_STEP,
    DISPATCH_LEAD_TIME_MINUTES,
    HIGH_LOAD_THRESHOLD,
    LOW_LOAD_THRESHOLD,
    MIN_CONFIDENCE_SCORE,
    RESERVE_COEF,
    UTILIZATION_THRESHOLD,
    VEHICLE_CAPACITY_CONTAINERS,
)


class DispatchService:
    def __init__(self) -> None:
        self.capacity = VEHICLE_CAPACITY_CONTAINERS
        self.reserve_coef = RESERVE_COEF
        self.lead_time_minutes = DISPATCH_LEAD_TIME_MINUTES
        self.recommendations_by_run: dict[str, list[dict[str, Any]]] = {}
        self.confirmations: list[dict[str, Any]] = []
        self.overrides: list[dict[str, Any]] = []

    def calculate_dispatch_plan(self, forecasts: pd.DataFrame, run_id: str | None = None) -> dict[str, Any]:
        if forecasts.empty:
            return {"run_id": run_id, "recommendations": [], "warehouse_summary": []}

        df = forecasts.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "confidence_score" not in df.columns:
            df["confidence_score"] = 0.7
        if "confidence" not in df.columns:
            df["confidence"] = df["confidence_score"].apply(self._label_confidence)

        min_ts = df["timestamp"].min()
        df["step_ahead"] = ((df["timestamp"] - min_ts).dt.total_seconds() / 1800).round().astype(int) + 1

        recommendations = []
        for office_id, office_df in df.groupby("office_from_id", sort=False):
            for step_ahead, step_df in office_df.groupby("step_ahead", sort=False):
                predicted_volume = float(step_df["predicted_target_2h"].sum())
                confidence_score = float(step_df["confidence_score"].mean())
                confidence = self._label_confidence(confidence_score)

                if predicted_volume < LOW_LOAD_THRESHOLD:
                    continue

                vehicles_needed = self._calculate_vehicles_needed(predicted_volume)
                priority = self._define_priority(predicted_volume, step_ahead)
                reason = self._build_reason(predicted_volume, step_ahead, confidence)
                requires_confirmation = (
                    confidence_score < MIN_CONFIDENCE_SCORE or step_ahead > AUTO_APPROVE_MAX_STEP
                )
                recommendation_id = str(uuid4())

                recommendations.append(
                    {
                        "recommendation_id": recommendation_id,
                        "run_id": run_id,
                        "office_from_id": int(office_id),
                        "forecast_for_step": predicted_volume,
                        "forecast_timestamp": step_df["timestamp"].min().isoformat(),
                        "step_ahead": int(step_ahead),
                        "vehicles_needed": vehicles_needed,
                        "vehicle_capacity": self.capacity,
                        "reserve_coef": self.reserve_coef,
                        "priority": priority,
                        "confidence": confidence,
                        "confidence_score": round(confidence_score, 3),
                        "reason": reason,
                        "dispatch_at": (
                            step_df["timestamp"].min() - timedelta(minutes=self.lead_time_minutes)
                        ).isoformat(),
                        "decision_status": "needs_confirmation" if requires_confirmation else "auto_ready",
                        "source": "forecast_based",
                    }
                )

        recommendations.sort(key=lambda row: (row["step_ahead"], row["office_from_id"]))
        self.recommendations_by_run[run_id or "adhoc"] = recommendations

        warehouse_summary = self._build_warehouse_summary(recommendations)
        return {
            "run_id": run_id,
            "recommendations": recommendations,
            "warehouse_summary": warehouse_summary,
        }

    def get_recommendations(self, run_id: str | None = None) -> dict[str, Any]:
        if run_id:
            return {"run_id": run_id, "recommendations": self.recommendations_by_run.get(run_id, [])}

        if not self.recommendations_by_run:
            return {"run_id": None, "recommendations": []}

        latest_run_id = list(self.recommendations_by_run.keys())[-1]
        return {"run_id": latest_run_id, "recommendations": self.recommendations_by_run[latest_run_id]}

    def confirm_recommendation(self, recommendation_id: str, user_id: str | None = None) -> dict[str, Any]:
        recommendation = self._find_recommendation(recommendation_id)
        recommendation["decision_status"] = "confirmed"
        audit = {
            "recommendation_id": recommendation_id,
            "user_id": user_id or "dispatcher",
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.confirmations.append(audit)
        return {"status": "confirmed", "recommendation": recommendation, "audit": audit}

    def override_recommendation(
        self,
        recommendation_id: str,
        new_vehicles_needed: int,
        comment: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        recommendation = self._find_recommendation(recommendation_id)
        old_value = recommendation["vehicles_needed"]
        recommendation["vehicles_needed"] = int(new_vehicles_needed)
        recommendation["decision_status"] = "overridden"
        recommendation["override_comment"] = comment

        override_record = {
            "override_id": str(uuid4()),
            "recommendation_id": recommendation_id,
            "old_value": old_value,
            "new_value": int(new_vehicles_needed),
            "comment": comment,
            "user_id": user_id or "dispatcher",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.overrides.append(override_record)
        return {"status": "overridden", "recommendation": recommendation, "override": override_record}

    def _calculate_vehicles_needed(self, predicted_volume: float) -> int:
        raw_vehicles = (predicted_volume * self.reserve_coef) / self.capacity
        vehicles = int(np.ceil(raw_vehicles))
        vehicles = max(vehicles, 1)

        utilization = predicted_volume / max(vehicles * self.capacity, 1e-9)
        if utilization < UTILIZATION_THRESHOLD and vehicles > 1:
            vehicles -= 1
        return vehicles

    def _define_priority(self, predicted_volume: float, step_ahead: int) -> str:
        if step_ahead <= 2 or predicted_volume >= HIGH_LOAD_THRESHOLD:
            return "high"
        if step_ahead <= 4:
            return "medium"
        return "low"

    def _build_reason(self, predicted_volume: float, step_ahead: int, confidence: str) -> str:
        if confidence == "low":
            return (
                f"Прогноз {predicted_volume:.1f} емкостей на шаг {step_ahead}; "
                "нужна проверка диспетчером из-за низкой уверенности."
            )
        if step_ahead <= AUTO_APPROVE_MAX_STEP:
            return (
                f"Ожидается рост нагрузки до {predicted_volume:.1f} емкостей "
                f"через {step_ahead * 30} минут."
            )
        return (
            f"Дальний горизонт: {predicted_volume:.1f} емкостей через {step_ahead * 30} минут, "
            "рекомендация для подготовки транспорта."
        )

    def _build_warehouse_summary(self, recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not recommendations:
            return []

        df = pd.DataFrame(recommendations)
        priority_rank = {"low": 1, "medium": 2, "high": 3}
        rank_to_priority = {value: key for key, value in priority_rank.items()}
        df["priority_rank"] = df["priority"].map(priority_rank).fillna(1)
        summary = (
            df.groupby("office_from_id", as_index=False)
            .agg(
                total_vehicles_needed=("vehicles_needed", "sum"),
                max_priority_rank=("priority_rank", "max"),
                max_forecast=("forecast_for_step", "max"),
                recommendations_count=("recommendation_id", "count"),
            )
            .sort_values(["total_vehicles_needed", "office_from_id"], ascending=[False, True])
        )
        summary["max_priority"] = summary["max_priority_rank"].map(rank_to_priority)
        summary = summary.drop(columns=["max_priority_rank"])
        return summary.to_dict(orient="records")

    def _find_recommendation(self, recommendation_id: str) -> dict[str, Any]:
        for recommendations in self.recommendations_by_run.values():
            for recommendation in recommendations:
                if recommendation["recommendation_id"] == recommendation_id:
                    return recommendation
        raise KeyError(f"Recommendation not found: {recommendation_id}")

    def _label_confidence(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"
