import pandas as pd
from models.stub import ForecastModelStub

class ForecastingService:
    def __init__(self):
        self.model = ForecastModelStub()

    def generate_forecast(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # Нормализация типов
        raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
        raw_data = raw_data.sort_values(["office_from_id", "route_id", "timestamp"])
        
        # Вызов модели
        preds = self.model.predict(raw_data)
        
        result = raw_data[["route_id", "office_from_id", "timestamp"]].copy()
        result["predicted_target_2h"] = preds.values
        result["confidence"] = "high"  # Упрощенно: можно добавить на основе дисперсии модели
        return result