# Отчёт по модели LightGBM v2 (+ погода)

## 1. Постановка эксперимента
- Train: 2025-01-01 … 2025-09-30
- Valid: 2025-10-01 … 2025-10-31
- Модель: LightGBM LGBMRegressor (objective=`regression_l1`)
- Внешние признаки (погода): да
- Исключённые признаки: ['month', 'dayofyear', 'weekofyear', 'day']
- Свежее окно уровня: 8 нед.
- Число признаков: 28
- Best iteration (early stopping): 769

## 2. Гиперпараметры
| Параметр | Значение |
|---|---|
| `boosting_type` | gbdt |
| `class_weight` | None |
| `colsample_bytree` | 0.9 |
| `importance_type` | split |
| `learning_rate` | 0.05 |
| `max_depth` | -1 |
| `min_child_samples` | 20 |
| `min_child_weight` | 0.001 |
| `min_split_gain` | 0.0 |
| `n_estimators` | 2000 |
| `n_jobs` | -1 |
| `num_leaves` | 31 |
| `objective` | regression_l1 |
| `random_state` | 42 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 0.1 |
| `subsample` | 0.9 |
| `subsample_for_bin` | 200000 |
| `subsample_freq` | 1 |
| `verbose` | -1 |

## 3. Метрики на октябрьской валидации
| Вариант | WAPE | WAPE-score | MAE |
|---|---|---|---|
| LightGBM (+постобработка) | 0.0928 | **0.9072** | 91.54 |
| Блендинг LGBM + медиана(4 нед) | 0.1032 | **0.8968** | 101.85 |

## 4. Топ-20 признаков по важности
| # | Признак | Важность (split) |
|---|---|---|
| 1 | `days_to_holiday` | 2228 |
| 2 | `hist_rhd_mean` | 2186 |
| 3 | `days_after_holiday` | 1635 |
| 4 | `temperature_2m` | 1632 |
| 5 | `hour` | 1510 |
| 6 | `hist_rd_med` | 1202 |
| 7 | `relative_humidity_2m` | 1157 |
| 8 | `wind_speed_10m` | 1090 |
| 9 | `route` | 999 |
| 10 | `hist_rhe_mean` | 954 |
| 11 | `snow_depth` | 949 |
| 12 | `hist_rhd_med` | 922 |
| 13 | `trend_ratio` | 803 |
| 14 | `recent_rh_mean` | 791 |
| 15 | `recent_rhd_med` | 677 |
| 16 | `dow` | 627 |
| 17 | `hist_rh_mean` | 616 |
| 18 | `is_weekend` | 560 |
| 19 | `cloud_cover` | 536 |
| 20 | `hist_rh_med` | 481 |

## 5. Вывод
- WAPE-score модели: **0.9072**, блендинга: **0.8968**.
- Порог baseline (~0.48) и целевой порог 0.85 достигнуты.
