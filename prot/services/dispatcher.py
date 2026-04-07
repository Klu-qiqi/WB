import pandas as pd
from datetime import timedelta
from config import (
    VEHICLE_CAPACITY_CONTAINERS,
    DISPATCH_LEAD_TIME_HOURS,
    UTILIZATION_THRESHOLD
)

class DispatchService:
    def __init__(self):
        self.capacity = VEHICLE_CAPACITY_CONTAINERS
        self.lead_time = DISPATCH_LEAD_TIME_HOURS

    def calculate_dispatch_plan(self, forecasts: pd.DataFrame) -> list[dict]:
        plan = []
        for _, row in forecasts.iterrows():
            pred = row["predicted_target_2h"]
            if pred <= 0.5:
                continue

            # Расчет машин с учетом порога утилизации
            raw_vehicles = pred / self.capacity
            vehicles = int(np.ceil(raw_vehicles))
            
            utilization = pred / (vehicles * self.capacity)
            if utilization < UTILIZATION_THRESHOLD and vehicles > 1:
                vehicles -= 1  # Консолидируем отгрузку, если утилизация низкая

            dispatch_time = pd.Timestamp(row["timestamp"]) + timedelta(hours=self.lead_time)
            
            plan.append({
                "route_id": row["route_id"],
                "office_from_id": row["office_from_id"],
                "predicted_containers_2h": round(pred, 2),
                "vehicles_required": vehicles,
                "dispatch_at": dispatch_time.isoformat(),
                "status": "scheduled",
                "source": "forecast_based"
            })
        return plan