# Бизнес-допущения и параметры системы
VEHICLE_CAPACITY_CONTAINERS = 10  # Вместимость одного ТС в емкостях (target_2h)
DISPATCH_LEAD_TIME_HOURS = 1.0    # Время на подготовку и подачу ТС к складу
FORECAST_UPDATE_INTERVAL_MINUTES = 30  # Частота обновления прогноза
PLANNING_HORIZON_HOURS = 4.0      # Горизонт, на котором уверены в прогнозе
UTILIZATION_THRESHOLD = 0.8       # Мин. заполняемость для оправданности вызова ТС
DATA_TIMEZONE = "Europe/Moscow"