from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, LSTM
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorFlow is required for this script. Install it with `pip install tensorflow`."
    ) from exc


@dataclass
class Config:
    data_path: Path = Path("/Users/lubovklepcova/Desktop/Siriusly/ML/wb/Dd2WPGKz")
    train_path: str = "train_team_track.parquet"
    test_path: str = "test_team_track.parquet"
    target_col: str = "target_2h"
    forecast_points: int = 10
    window: int = 48
    valid_ratio: float = 0.2
    random_state: int = 42
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    lstm_units: int = 128
    dropout: float = 0.2
    min_history: int = 120


class WapePlusRBias:
    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        den = np.maximum(np.abs(y_true).sum(), 1e-12)
        wape = np.abs(y_pred - y_true).sum() / den
        rbias = np.abs((y_pred.sum() - y_true.sum()) / den)
        return float(wape + rbias)


def build_supervised(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build route-wise sliding windows:
    X: [samples, window, features]
    y: [samples, forecast_points]
    ts: timestamp of window end (for temporal split)
    """
    feature_cols = [f"status_{i}" for i in range(1, 9)] + [cfg.target_col]
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    ts_blocks: list[np.ndarray] = []

    for _, grp in df.groupby("route_id", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < cfg.min_history:
            continue

        vals = grp[feature_cols].to_numpy(dtype=np.float32)
        tgt = grp[cfg.target_col].to_numpy(dtype=np.float32)
        ts = grp["timestamp"].to_numpy()

        max_start = len(grp) - cfg.window - cfg.forecast_points + 1
        if max_start <= 0:
            continue

        for start in range(max_start):
            end = start + cfg.window
            horizon_end = end + cfg.forecast_points
            x_blocks.append(vals[start:end])
            y_blocks.append(tgt[end:horizon_end])
            ts_blocks.append(np.array([ts[end - 1]], dtype="datetime64[ns]"))

    if not x_blocks:
        raise ValueError("No supervised samples built. Reduce window/min_history.")

    X = np.stack(x_blocks, axis=0)
    y = np.stack(y_blocks, axis=0)
    ts = np.concatenate(ts_blocks, axis=0)
    return X, y, ts


def build_test_windows(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = [f"status_{i}" for i in range(1, 9)] + [cfg.target_col]
    route_ids: list[int] = []
    x_blocks: list[np.ndarray] = []

    for route_id, grp in df.groupby("route_id", sort=False):
        grp = grp.sort_values("timestamp")
        if len(grp) < cfg.window:
            continue
        x_blocks.append(grp[feature_cols].to_numpy(dtype=np.float32)[-cfg.window:])
        route_ids.append(route_id)

    X_test = np.stack(x_blocks, axis=0)
    return X_test, np.array(route_ids, dtype=int)


def scale_3d(
    X_fit: np.ndarray, X_valid: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_fit, seq_len, n_feat = X_fit.shape
    scaler = StandardScaler()
    scaler.fit(X_fit.reshape(-1, n_feat))

    X_fit_s = scaler.transform(X_fit.reshape(-1, n_feat)).reshape(n_fit, seq_len, n_feat)
    X_valid_s = scaler.transform(X_valid.reshape(-1, n_feat)).reshape(X_valid.shape[0], seq_len, n_feat)
    X_test_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape[0], seq_len, n_feat)
    return X_fit_s, X_valid_s, X_test_s


def build_model(seq_len: int, n_feat: int, horizon: int, cfg: Config) -> tf.keras.Model:
    model = Sequential(
        [
            LSTM(cfg.lstm_units, input_shape=(seq_len, n_feat), return_sequences=False),
            Dropout(cfg.dropout),
            Dense(64, activation="relu"),
            Dense(horizon, activation="linear"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="mae",
    )
    return model


def main() -> None:
    cfg = Config()
    np.random.seed(cfg.random_state)
    tf.random.set_seed(cfg.random_state)

    train_df = pd.read_parquet(cfg.data_path / cfg.train_path)
    test_df = pd.read_parquet(cfg.data_path / cfg.test_path)
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    train_df = train_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    test_df = test_df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    X, y, ts = build_supervised(train_df, cfg)
    split_point = pd.Series(ts).quantile(1 - cfg.valid_ratio)
    fit_mask = ts <= split_point
    valid_mask = ts > split_point

    X_fit, y_fit = X[fit_mask], y[fit_mask]
    X_valid, y_valid = X[valid_mask], y[valid_mask]

    X_test_raw, test_route_ids = build_test_windows(train_df, cfg)
    X_fit, X_valid, X_test = scale_3d(X_fit, X_valid, X_test_raw)

    model = build_model(X_fit.shape[1], X_fit.shape[2], cfg.forecast_points, cfg)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_fit,
        y_fit,
        validation_data=(X_valid, y_valid),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    fit_pred = np.maximum(model.predict(X_fit, verbose=0), 0.0)
    valid_pred = np.maximum(model.predict(X_valid, verbose=0), 0.0)
    print("fit WAPE+|RBias|:", round(WapePlusRBias.calculate(y_fit, fit_pred), 6))
    print("valid WAPE+|RBias|:", round(WapePlusRBias.calculate(y_valid, valid_pred), 6))

    test_pred = np.maximum(model.predict(X_test, verbose=0), 0.0)
    inference_ts = train_df["timestamp"].max()

    rows = []
    for i, route_id in enumerate(test_route_ids):
        for step in range(1, cfg.forecast_points + 1):
            rows.append(
                {
                    "route_id": int(route_id),
                    "timestamp": inference_ts + pd.Timedelta(minutes=30 * step),
                    "y_pred": float(test_pred[i, step - 1]),
                }
            )
    forecast_df = pd.DataFrame(rows)

    submission_df = test_df.merge(forecast_df, on=["route_id", "timestamp"], how="left")[["id", "y_pred"]]
    if submission_df["y_pred"].isna().any():
        # Safety fallback for routes with short history.
        route_mean = train_df.groupby("route_id", as_index=True)[cfg.target_col].mean()
        submission_df = test_df[["id", "route_id"]].join(
            test_df["route_id"].map(route_mean).rename("y_pred"),
            how="left",
        )[["id", "y_pred"]]
        submission_df["y_pred"] = submission_df["y_pred"].fillna(train_df[cfg.target_col].mean())

    out_path = Path("submission_team_rnn.csv")
    submission_df.to_csv(out_path, index=False)
    print(f"submission saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
