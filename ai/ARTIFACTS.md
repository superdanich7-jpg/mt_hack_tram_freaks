# ARTIFACTS.md — что ML отдаёт команде

Капитан/ML обязан передать команде следующие артефакты, чтобы они
могли работать без него.

## 1. `submissions/submission.csv`

Финальный файл для загрузки в «Data Science».

## 2. `submissions/forecast.csv`

Полный прогноз в удобном виде для веб-сервиса.

Колонки:
route;date;hour;prediction

text

Тот же формат, что сабмит. Плюс, при желании, дополнительные поля:

- `prediction_lower`, `prediction_upper` — границы (если есть).
- `is_holiday`, `temperature_2m` — контекст.

## 3. `models/`

- `lgbm_model.txt` (или аналог).
- `feature_importance.csv`.
- `feature_list.json`.

## 4. `reports/`

- `eda.md` — что в данных.
- `baseline.md` — метрики baseline.
- `model_v1.md` — первая модель.
- `model_v2.md` — с внешними данными.
- `validation.md` — финальные метрики на октябре.

## 5. `README_ML.md`

Пошагово:

1. Как установить зависимости.
2. Как обучить модель.
3. Как получить сабмит.
4. Как проверить сабмит.
5. Где лежат артефакты.

## 6. `api_spec.md`

Контракт для backend:

- `GET /routes` — список маршрутов.
- `GET /forecast?route=1&date=2025-11-01` — прогноз на день.
- `GET /forecast/week?route=1&start=2025-11-01` — прогноз на неделю.
- `GET /forecast/month?route=1&month=2025-11` — прогноз на месяц.
- `GET /anomalies` — маршруты с аномалиями.

Формат ответа: JSON.

## 7. Docker-образ для ML (опционально)

Если сервис будет запускать модель онлайн:

- `Dockerfile.ml`.
- Модель внутри.
- Endpoint `POST /predict`.

## 8. Демо-ноутбук

`notebooks/03_demo.ipynb` — быстро показать, как всё работает.