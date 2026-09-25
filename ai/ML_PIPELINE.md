# ML_PIPELINE.md — пошаговый пайплайн

Порядок действий. Каждый шаг — отдельный модуль в `src/`.

## Шаг 1. Загрузка и очистка данных

`src/data.py`

- Читать `labels_day_train.csv` и `labels_day_test.csv` (sep=`;`).
- Парсить `date` как `datetime`.
- Привести `route`, `hour`, `boardings` к int.
- Объединить в один DataFrame.
- Построить полную сетку `route × date × hour`.
- Заполнить пропуски `boardings=0`.
- Отдельно сохранить список реальных маршрутов: `[1,7,11,12,17,25,26,28,50]`.
- Опционально: пометить часы 2–4 как «технологические».
- Сохранить в `data/processed/boardings.parquet`.

## Шаг 2. EDA

`notebooks/01_eda.ipynb` → отчёт `reports/eda.md`

- Распределение по часам (сутки).
- Распределение по дням недели.
- Сравнение месячных профилей.
- Аномалии: провалы, всплески.
- Тепловая карта `route × hour`.
- Автокорреляции.

## Шаг 3. Baseline

`src/baseline.py`

- Наивный прогноз: `median(boardings)` по `route + hour + dow`
  за последние 4 недели.
- Посчитать WAPE на октябре.
- Сохранить результат в `reports/baseline.md`.

## Шаг 4. Первая модель

`src/train.py`, `src/features.py`, `src/predict.py`

- Признаки: см. `ai/FEATURES.md`.
- Модель: LightGBM (или CatBoost, HistGradientBoosting).
- Loss: `MAE` / `regression_l1`.
- Валидация на октябре.
- Если бьёт baseline → сохраняем модель в `models/`.

## Шаг 5. Внешние данные

`src/external/weather.py`, `src/external/holidays.py`

- Погода Open-Meteo Archive за ноябрь–декабрь 2025.
- Праздники РФ: 4 ноября, 31 декабря (короткий день).
- Плюс — школьные каникулы, если найдёшь.
- Приджойнить по `date + hour` (погода) и `date` (праздники).
- Переобучить модель.
- Сравнить с моделью без внешних данных на октябре.

## Шаг 6. Финальный прогноз

`src/predict.py`

- Обучить на 2025-01-01 … 2025-10-31.
- Сформировать сетку ноябрь–декабрь.
- Предсказать.
- Округлить, clip ≥ 0.
- route=5 → 0.
- Сохранить `submissions/forecast.csv` и `submissions/submission.csv`.

## Шаг 7. Валидация сабмита

`src/validate_submission.py`

- Проверить 14 640 строк.
- Все маршруты, все даты, все часы.
- Нет NaN.
- `prediction` целые ≥ 0.
- Разделитель и формат даты совпадают с `test_submission.csv`.

## Шаг 8. Артефакты для команды

См. `ai/ARTIFACTS.md` и `ai/HANDOFF.md`.

## Воспроизводимость

- Один скрипт: `python -m src.main` — от данных до сабмита.
- Или Makefile: `make train`, `make predict`, `make validate`.
- Seed фиксирован.
- Версии зависимостей — в `requirements.txt`.