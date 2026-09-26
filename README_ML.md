# README_ML.md — ML-часть: как запустить и что получилось

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

## 1. Установка зависимостей

```bash
python -m pip install -r requirements.txt
```

Ключевые пакеты: `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `pyarrow`,
`requests`, `pytest`. Для экспериментов V3 дополнительно `catboost`,
`xgboost`, `optuna`, `torch`, `statsmodels`, `openpyxl`.

Требуется Python 3.10+ (проверено на 3.14, pandas 3.0.3, lightgbm 4.7.0).


## 2. Данные

Исходные данные — `labels/labels_day_train.csv` (янв–авг) и
`labels/labels_day_test.csv` (сен–окт), разделитель `;`.

```bash
python -m src.data          # -> data/processed/boardings.parquet (9 маршрутов × 304 дня × 24 ч)
```

Внешние данные (уже закэшированы в `data/external/`, при необходимости обновить):

```bash
python -m src.external.weather    # Open-Meteo Archive 2025, почасово -> data/external/weather.csv
python -m src.external.holidays   # праздники РФ -> data/external/holidays.csv
```

> Замечание: `src/external/weather.py` отключает проверку SSL-сертификата —
> на этой машине локальный Windows trust store устарел. Решение зафиксировано
> в `reports/decisions.md`.

## 3. Обучение модели

```bash
# Базовая модель: полный календарь, без погоды
python -m src.train --out models/lgbm_v1.txt --report reports/model_v1.md

# Финальная модель: без сезонного календаря + погода, окно уровня 8 недель
python -m src.train \
  --exclude "month,dayofyear,weekofyear,day" \
  --weather --recent-weeks 8 \
  --out models/lgbm_v2.txt --report reports/model_v2.md \
  --importance-csv reports/feature_importance.csv
```

Полезные флаги `src.train`:

| Флаг | Смысл |
|---|---|
| `--weather` | добавить погодные признаки |
| `--exclude a,b,c` | исключить признаки (абляции) |
| `--recent-weeks N` | длина «свежего» окна уровня, по умолчанию 4 |
| `--valid-start YYYY-MM-DD` | начало валидации; `2025-09-01` включает 2-месячный прокси |
| `--leaves N`, `--lr F` | переопределить гиперпараметры |
| `--ensemble` | сравнить v1, v2 и их среднее на сплите |

Валидация по умолчанию: train `2025-01-01…2025-09-30`, valid `2025-10-01…2025-10-31`.

## 3.1. Эксперименты V3 (воспроизводимые)

Каждый этап кэширует результат в `reports/v3/`, поэтому отчёт можно
пересобрать без повторного обучения.

```bash
# Абляция групп признаков на трёх временных фолдах (~3 мин)
python -m src.experiment_v3 --stage ablate

# Подбор гиперпараметров Optuna на тестовом фолде (~15 мин, CatBoost самый долгий)
python -m src.experiment_v3 --stage tune

# Честная проверка тюнинга: подбор на 2025-09, оценка на невиданных фолдах (~10 мин)
python -m src.experiment_v3 --stage tune-holdout --holdout-trials 15

# Сравнение архитектур: LightGBM / CatBoost / XGBoost / LSTM+attn / ETS (~15 мин)
python -m src.experiment_v3 --stage models --extra cross_route

# Блендинг и стэкинг с out-of-time весами (секунды, читает models/preds)
python -m src.experiment_v3 --stage ensemble

# Пересборка reports/final_model_analysis.md
python -m src.experiment_v3 --stage report
```

Схема фолдов — в `src/cv.py`: два валидационных окна по 14 дней
(2025-08, 2025-09) и тестовый месяц (2025-10). Обучение всегда строго
раньше валидации, агрегаты признаков считаются только по `date < cutoff`.

Дополнительные модули:

| Модуль | Назначение |
|---|---|
| `src/cv.py` | скользящие временные фолды |
| `src/models_zoo.py` | единый интерфейс для LightGBM / CatBoost / XGBoost |
| `src/tune.py` | Optuna (TPE) и перебор кандидатов |
| `src/nn_models.py` | LSTM + self-attention (PyTorch) |
| `src/stats_models.py` | ETS, SARIMAX, seasonal naive |
| `src/ensemble.py` | simple / weighted average, стэкинг |
| `src/validations_eda.py` | разбор `Хакатон_пример_валидаций.xlsx` |
| `src/report_v3.py` | генерация `reports/final_model_analysis.md` |

Проверить, годится ли файл валидаций как источник признаков:

```bash
python -m src.validations_eda
```

Ответ: **нет** — 20 строк, дата 2026-07-27, пересечения с разметкой 2025 года
не существует. Подробности в разделе 4 отчёта.

## 4. Получение сабмита

```bash
python -m src.predict            # финальная модель (LightGBM + cross_route)
python -m src.predict --ensemble # эксперимент: ансамбль GBM (проигрывает, см. отчёт)
```

Шаги внутри: early stopping на октябре определяет число деревьев (1625),
модель переобучается на всей истории `2025-01-01…2025-10-31` с гиперпараметрами
из `config.LGBM_TUNED_PARAMS`, строится сетка 10 маршрутов × 61 день × 24 часа,
`route=5` обнуляется, значения округляются и клипаются по нулю.


Результаты:

- `submissions/forecast.csv` — полный прогноз, `route;date;hour;prediction`.
- `submissions/submission.csv` — файл для загрузки (порядок строк и формат
  даты берутся из реального `test_submission.csv`).

## 5. Проверка сабмита

```bash
python -m src.validate_submission
```

Проверяются 24 условия из `ai/SUBMISSION.md`: 14 640 строк, все 10 маршрутов,
61 дата, 24 часа на пару `(route, date)`, отсутствие NaN и дубликатов,
целые `prediction ≥ 0`, `route=5` = 0, разделитель `;`, формат даты и порядок
строк как в `test_submission.csv`. Завершается словом `OK` или списком `FAIL`.

## 6. Полный цикл одной командой

```bash
python -m src.main                  # данные -> обучение v1/v2 -> прогноз -> валидация
python -m src.main --skip-experiments   # только финальный прогноз + валидация
```

Через Makefile:

```bash
make data        # подготовка parquet
make train       # v1 + v2
make predict     # submissions/*.csv
make validate    # проверка формата
make test        # 22 pytest-теста
make all         # data + train + predict + validate
```

## 7. Где лежат артефакты

| Путь | Что это |
|---|---|
| `submissions/submission.csv` | финальный файл для проверяющей системы |
| `submissions/forecast.csv` | прогноз для веб-сервиса |
| `models/lgbm_final.txt` | финальная обученная модель (Booster) |
| `models/lgbm_v1.txt`, `models/lgbm_v2.txt` | модели валидационных экспериментов |
| `models/feature_importance.csv`, `models/feature_list.json` | важность и список признаков |
| `reports/eda.md` | разведка данных |
| `reports/baseline.md` | метрики baseline |
| `reports/model_v1.md`, `reports/model_v2.md` | эксперименты моделей |
| `reports/validation.md` | итоговые метрики и сравнение |
| `reports/decisions.md` | журнал всех принятых решений |
| `api_spec.md` | контракт для backend/веб-сервиса |
| `tests/` | pytest: данные, признаки, формат сабмита |

## 8. Конфигурация

Все пути, даты и гиперпараметры — в `src/config.py` (`RANDOM_SEED = 42`,
`LGBM_PARAMS`, окна дат). «Магических» чисел в коде нет.

## 9. Что важно знать про данные и модель

- ⚠️ **Формат даты:** заметка в `ai/SUBMISSION.md` про `DD.MM.YYYY` ошибочна.
  Реальный `test_submission.csv` содержит `1;2025-11-01;0;349` (то есть
  `YYYY-MM-DD`), и исходное ТЗ в `docs/qa/prompt.txt` тоже требует
  `YYYY-MM-DD` и советует сохранить структуру шаблона. `src/predict.py`
  собирает сабмит merge'ом на строки шаблона, поэтому формат и порядок строк
  гарантированно совпадают с ним, что бы ни было написано в документации.
- `route=5` нет в разметке (обучать нельзя) — в сабмите нули, потеря ≈0.006 метрики.
- Значения `prediction` в `test_submission.csv` — дамми: зависят только от
  маршрута и блока часов, одинаковы для всех 61 дня. Их оценка на октябре
  даёт ровно 0.4776, то есть заявленный бенчмарк 0.48.
- Главный риск — уровень пассажиропотока в ноябре–декабре (будущее не наблюдаемо).
  Поэтому признаки-«сезон года» (`month`, `dayofyear`, `weekofyear`, `day`)
  исключены: деревья не умеют экстраполировать за пределы обучающего диапазона,
  и на 2-месячном прокси-сплите они ухудшали WAPE-score на 0.03–0.09.
  Уровень заякорен на медианы `(route, hour, dow)` за последние 8 недель истории.
