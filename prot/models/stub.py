import pandas as pd
import numpy as np

class ForecastModelStub:
    """
    Заглушка прогнозной модели.
    ЗАМЕНИТЬ: self.model = joblib.load("your_model.pkl")
    """
    def __init__(self):
        self.is_loaded = False
        # self.model = joblib.load("model.pkl")  # Раскомментировать при подключении реальной модели
        # self.is_loaded = True

    def predict(self, data: pd.DataFrame) -> pd.Series:
        if data.empty:
            return pd.Series(dtype=float)
        
        # Пример простой эвристики на основе статусов
        # В реальной системе здесь будет model.predict(data)
        status_cols = [c for c in data.columns if c.startswith("status_")]
        # Суммарный поток за 30 мин -> грубая оценка за 2 часа (коэффициент ~4)
        preds = data[status_cols].sum(axis=1) * 0.35
        return preds.clip(lower=0).round(2)