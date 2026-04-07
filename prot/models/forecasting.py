from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import (
    CHAMPION_MODEL_NAME,
    CHAMPION_MODEL_VERSION,
    DATA_PATH_TRAIN,
    FALLBACK_MODEL_NAME,
    FALLBACK_MODEL_VERSION,
    MODEL_RANDOM_STATE,
)

TARGET_COL = "target_2h"
KEY_COLS = ["route_id", "timestamp"]
LAGS = [1, 2, 4, 8, 16, 48]
ROLL_WINDOWS = [4, 12, 48]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour.astype("int16")
    out["dow"] = out["timestamp"].dt.dayofweek.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).astype("float32")
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7).astype("float32")
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7).astype("float32")
    return out


def add_target_lag_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(KEY_COLS).copy()
    grouped_target = out.groupby("route_id", sort=False)[TARGET_COL]

    for lag in LAGS:
        out[f"y_lag_{lag}"] = grouped_target.shift(lag).astype("float32")

    shifted = grouped_target.shift(1)
    for window in ROLL_WINDOWS:
        out[f"y_roll_mean_{window}"] = (
            shifted.groupby(out["route_id"], sort=False)
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
            .astype("float32")
        )
        out[f"y_roll_std_{window}"] = (
            shifted.groupby(out["route_id"], sort=False)
            .rolling(window)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .astype("float32")
        )

    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(df)
    out = add_target_lag_roll_features(out)
    return out


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else np.nan


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def make_row_features_from_history(route_id: int, ts: pd.Timestamp, history: list[float]) -> dict[str, Any]:
    hour = ts.hour
    dow = ts.dayofweek

    row = {
        "route_id": route_id,
        "timestamp": ts,
        "hour": hour,
        "dow": dow,
        "is_weekend": int(dow >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
    }

    for lag in LAGS:
        row[f"y_lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan

    for window in ROLL_WINDOWS:
        window_values = history[-window:] if history else []
        row[f"y_roll_mean_{window}"] = _safe_mean(window_values)
        row[f"y_roll_std_{window}"] = _safe_std(window_values)

    return row


def _resolve_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    prot_relative = Path(__file__).resolve().parents[1] / path_str
    if prot_relative.exists():
        return prot_relative

    repo_relative = Path(__file__).resolve().parents[2] / path_str
    if repo_relative.exists():
        return repo_relative

    raise FileNotFoundError(f"Dataset not found: {path_str}")


@dataclass
class ModelDescriptor:
    name: str
    version: str
    model_type: str
    is_champion: bool
    mode: str
    status: str
    score_offline: float | None = None
    details: str | None = None


class HistoryOnlyBaselineModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.array([], dtype=float)

        lag_cols = [col for col in X.columns if col.startswith("y_lag_")]
        roll_mean_cols = [col for col in X.columns if col.startswith("y_roll_mean_")]

        lag_part = X[lag_cols].mean(axis=1) if lag_cols else 0.0
        roll_part = X[roll_mean_cols].mean(axis=1) if roll_mean_cols else 0.0
        preds = 0.65 * lag_part + 0.35 * roll_part
        return np.clip(np.nan_to_num(preds.to_numpy(dtype=float), nan=0.0), 0.0, None)


class ForecastModelRegistry:
    def __init__(self) -> None:
        self.feature_cols: list[str] = []
        self.history_df: pd.DataFrame | None = None
        self.champion_model: RandomForestRegressor | None = None
        self.fallback_model = HistoryOnlyBaselineModel()
        self.active_model_name = FALLBACK_MODEL_NAME
        self.active_model_version = FALLBACK_MODEL_VERSION
        self.status = "fallback"
        self.initialization_error: str | None = None
        self.offline_scores = {
            CHAMPION_MODEL_NAME: None,
            FALLBACK_MODEL_NAME: None,
        }
        self._bootstrap()

    def _bootstrap(self) -> None:
        try:
            train_path = _resolve_data_path(DATA_PATH_TRAIN)
            train_df = pd.read_parquet(train_path)
            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            train_df = train_df.sort_values(KEY_COLS).reset_index(drop=True)
            self.history_df = train_df.copy()

            feat_full = build_feature_table(train_df)
            blocked_cols = {"office_from_id", TARGET_COL, "timestamp"}
            blocked_cols.update(c for c in feat_full.columns if c.startswith("status_"))
            self.feature_cols = [c for c in feat_full.columns if c not in blocked_cols]

            model_df = feat_full.dropna(subset=self.feature_cols + [TARGET_COL]).copy()
            X_full = model_df[self.feature_cols]
            y_full = model_df[TARGET_COL].values

            champion = RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=MODEL_RANDOM_STATE,
            )
            champion.fit(X_full, y_full)

            self.champion_model = champion
            self.active_model_name = CHAMPION_MODEL_NAME
            self.active_model_version = CHAMPION_MODEL_VERSION
            self.status = "champion"
            self.initialization_error = None
        except Exception as exc:
            self.champion_model = None
            self.active_model_name = FALLBACK_MODEL_NAME
            self.active_model_version = FALLBACK_MODEL_VERSION
            self.status = "fallback"
            self.initialization_error = str(exc)

    def get_registry_status(self) -> dict[str, Any]:
        models = [
            {
                "model_name": CHAMPION_MODEL_NAME,
                "version": CHAMPION_MODEL_VERSION,
                "type": "rf",
                "score_offline": self.offline_scores[CHAMPION_MODEL_NAME],
                "is_champion": self.champion_model is not None,
                "status": "ready" if self.champion_model is not None else "unavailable",
            },
            {
                "model_name": FALLBACK_MODEL_NAME,
                "version": FALLBACK_MODEL_VERSION,
                "type": "baseline",
                "score_offline": self.offline_scores[FALLBACK_MODEL_NAME],
                "is_champion": False,
                "status": "ready",
            },
        ]
        return {
            "active_champion": self.active_model_name,
            "active_version": self.active_model_version,
            "mode": self.status,
            "models": models,
            "initialization_error": self.initialization_error,
        }

    def predict_recursive(
        self,
        request_df: pd.DataFrame,
        use_enriched_inference: bool = False,
    ) -> tuple[pd.DataFrame, ModelDescriptor]:
        if request_df.empty:
            descriptor = self._choose_model_descriptor(use_enriched_inference)
            return pd.DataFrame(columns=["route_id", "office_from_id", "timestamp", "predicted_target_2h"]), descriptor

        if self.history_df is None:
            raise RuntimeError("Historical train data is not loaded, inference is unavailable.")

        request_df = request_df.copy()
        request_df["timestamp"] = pd.to_datetime(request_df["timestamp"])
        request_df = request_df.sort_values(["timestamp", "route_id"]).reset_index(drop=True)

        model, descriptor = self._choose_model(use_enriched_inference)

        history_by_route = {}
        office_by_route = {}
        for route_id, group in self.history_df.sort_values(KEY_COLS).groupby("route_id", sort=False):
            history_by_route[route_id] = group[TARGET_COL].tolist()
            office_by_route[route_id] = group["office_from_id"].iloc[-1]

        predictions = []
        for ts in sorted(request_df["timestamp"].unique()):
            step_slice = request_df[request_df["timestamp"] == ts].copy()
            step_rows = []
            step_route_ids = []

            for _, row in step_slice.iterrows():
                route_id = row["route_id"]
                history = history_by_route.get(route_id, [])
                office_by_route[route_id] = row.get("office_from_id", office_by_route.get(route_id))
                step_rows.append(make_row_features_from_history(route_id, ts, history))
                step_route_ids.append(route_id)

            X_step = pd.DataFrame(step_rows).reindex(columns=self.feature_cols).fillna(0.0)
            y_hat = np.clip(model.predict(X_step), 0.0, None)

            for route_id, pred in zip(step_route_ids, y_hat):
                history_by_route.setdefault(route_id, []).append(float(pred))
                predictions.append(
                    {
                        "route_id": route_id,
                        "office_from_id": office_by_route.get(route_id),
                        "timestamp": ts,
                        "predicted_target_2h": float(pred),
                    }
                )

        pred_df = pd.DataFrame(predictions)
        return pred_df, descriptor

    def _choose_model_descriptor(self, use_enriched_inference: bool) -> ModelDescriptor:
        if self.champion_model is not None:
            return ModelDescriptor(
                name=CHAMPION_MODEL_NAME,
                version=CHAMPION_MODEL_VERSION,
                model_type="rf",
                is_champion=True,
                mode="enriched" if use_enriched_inference else "history_only",
                status="ready",
            )
        return ModelDescriptor(
            name=FALLBACK_MODEL_NAME,
            version=FALLBACK_MODEL_VERSION,
            model_type="baseline",
            is_champion=False,
            mode="history_only",
            status="fallback",
            details=self.initialization_error,
        )

    def _choose_model(self, use_enriched_inference: bool) -> tuple[Any, ModelDescriptor]:
        if self.champion_model is not None:
            return self.champion_model, self._choose_model_descriptor(use_enriched_inference)
        return self.fallback_model, self._choose_model_descriptor(False)
