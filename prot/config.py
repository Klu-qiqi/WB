import os
from pathlib import Path

# ML / model registry
CHAMPION_MODEL_NAME = "random_forest_champion"
CHAMPION_MODEL_VERSION = "rf_v1"
FALLBACK_MODEL_NAME = "baseline_history_only"
FALLBACK_MODEL_VERSION = "baseline_v1"
MODEL_RANDOM_STATE = 42

# Data paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DATA_PATH_TRAIN = os.getenv("DATA_PATH_TRAIN", "Dd2WPGKz/train_team_track.parquet")
DATA_PATH_TEST = os.getenv("DATA_PATH_TEST", "Dd2WPGKz/test_team_track.parquet")

# Business assumptions
VEHICLE_CAPACITY_CONTAINERS = 10
RESERVE_COEF = 1.15
DISPATCH_LEAD_TIME_MINUTES = 60
FORECAST_UPDATE_INTERVAL_MINUTES = 30
PLANNING_HORIZON_STEPS = 10
AUTO_APPROVE_MAX_STEP = 4
LOW_LOAD_THRESHOLD = 0.5
HIGH_LOAD_THRESHOLD = 20.0
UTILIZATION_THRESHOLD = 0.8
MIN_CONFIDENCE_SCORE = 0.55
DATA_TIMEZONE = "Europe/Moscow"
