# ИИ-прогноз загрузки трамвайных маршрутов

Хакатон МТТ «ИИ-прогноз загрузки трамвайных маршрутов». Прогноз почасового
пассажиропотока по 9 трамвайным маршрутам Мосметро на горизонте
**ноябрь–декабрь 2025**.

## Текущий результат

| Метрика (тест = октябрь 2025, модель на данных до 30.09) | Значение |
|---|---|
| **WAPE-score** | **0.9082** |
| MAE, пассажиров/час | 90.59 |
| Бенчмарк организаторов (файл-шаблон) | 0.4776 |
| Сильнейший наивный baseline | 0.9040 |
| Сабмит | `submissions/submission.csv` — 14 640 строк, валидация 24/24 OK |

Модель: LightGBM (`regression_l1`), 31 признак, 1625 деревьев.

## Быстрый старт

```bash
python -m pip install -r requirements.txt
python -m src.main          # весь пайплайн: данные → обучение → сабмит → проверка
```

Или по шагам:

```bash
python -m src.data                  # -> data/processed/boardings.parquet
python -m src.external.weather      # погода Open-Meteo (нужна для погодных признаков)
python -m src.external.holidays     # календарь праздников РФ
python -m src.predict               # -> submissions/submission.csv + forecast.csv
python -m src.validate_submission   # проверка формата (24 условия)
python -m pytest tests -q           # 47 тестов
```

Если `data/external/weather.csv` отсутствует, пайплайн не падает: он
переключается на календарные признаки и пишет предупреждение в лог.

## Структура репозитория

```
src/                  ML-контур (Python)
  data.py             загрузка labels, построение сетки route × date × hour
  features.py         признаки; все агрегаты строго по date < cutoff
  cv.py               скользящие временные фолды
  models_zoo.py       LightGBM / CatBoost / XGBoost — единый интерфейс
  tune.py             Optuna (TPE)
  nn_models.py        LSTM + self-attention (PyTorch)
  stats_models.py     ETS, SARIMAX, seasonal naive
  ensemble.py         simple/weighted average, стэкинг
  experiment_v3.py    оркестрация всех экспериментов
  report_v3.py        генерация reports/final_model_analysis.md
  train.py            обучение и валидация
  predict.py          финальный сабмит
  validate_submission.py  проверка сабмита
  main.py             один запуск «от данных до сабмита»
submissions/          submission.csv (для загрузки) и forecast.csv (для сервиса)
models/               lgbm_final.txt, feature_list.json, feature_importance*.csv
reports/              отчёты: eda, baseline, model_v1, model_v2, validation,
                      final_model_analysis, decisions
api_spec.md           контракт REST API для backend
ai/                   правила и конвенции проекта
docs/qa/              исходное ТЗ и материалы QA
```

## Документация

| Документ | О чём |
|---|---|
| [`README_ML.md`](README_ML.md) | подробная инструкция по ML-части |
| [`reports/final_model_analysis.md`](reports/final_model_analysis.md) | полный разбор поиска модели: что пробовали, что отклонили и почему |
| [`reports/validation.md`](reports/validation.md) | метрики финального прогноза |
| [`reports/decisions.md`](reports/decisions.md) | журнал архитектурных решений |
| [`api_spec.md`](api_spec.md) | REST API для backend-сервиса |
| [`ai/ARTIFACTS.md`](ai/ARTIFACTS.md) | что команда получает от ML |

## Главный вывод по модели

V3 провёл полный поиск — новые группы признаков, CatBoost, XGBoost,
LSTM+attention, ETS/SARIMAX, Optuna, четыре варианта ансамблирования — и
**лишь два изменения прошли отбор**: перекрёстные (cross-route) признаки
и гиперпараметры, найденные Optuna на валидационном фолде 2025-09.
Все остальные варианты уступали одиночному LightGBM. Подробности и
числа — в `reports/final_model_analysis.md`.

Ключевой методологический момент: решения принимались по среднему на трёх
временных фолдах, а не по лучшему числу на одном месяце, поскольку
разброс между месяцами (≈0.017 WAPE-score) в 2–3 раза больше разницы
между конфигурациями (≈0.005).

## Область определения и ограничения

- Модель обучена на 9 маршрутах (1, 7, 11, 12, 17, 25, 26, 28, 50);
  `route=5` в сабмите заполняется нулями по условию задачи.
- Горизонт прогноза — до 2 месяцев без доступа к «свежим» данным,
  поэтому сезонные календарные признаки исключены: деревья не умеют
  экстраполировать их за пределы обучающего диапазона.
- Зависимость от внешних данных: Open-Meteo (фактическая погода) и
  календарь праздников РФ. Без погоды модель работает, но хуже.
- Перенос на другие маршруты/периоды потребует переобучения: подбор
  числа деревьев делается через early stopping на отложенном месяце.

Кратко: LightGBM (`regression_l1`) предсказывает почасовые посадки на
`route × date × hour` для 9 маршрутов; прогноз строится на ноябрь–декабрь 2025.

**Итог V3: WAPE-score = 0.9082 на октябрьском тесте (MAE 90.59) против 0.9072 у V2,
бенчмарка 0.4776 (файл-шаблон организаторов) и 0.9040 (сильнейший наивный baseline).**

Полный разбор поиска (признаки, CatBoost/XGBoost, LSTM+attention, ETS/SARIMAX,
Optuna, ансамбли) — в [`reports/final_model_analysis.md`](reports/final_model_analysis.md).
Коротко: из всего поиска отбор прошли **только два** изменения — перекрёстные
(cross-route) признаки и гиперпараметры, найденные Optuna на валидационном
фолде 2025-09. CatBoost, XGBoost, нейросеть, статистические модели и все
варианты ансамблирования оказались хуже одиночного LightGBM.

---
