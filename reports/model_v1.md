# Отчёт по модели LightGBM v1 (без погоды)

## 1. Постановка эксперимента
- Train: 2025-01-01 … 2025-09-30
- Valid: 2025-10-01 … 2025-10-31
- Модель: LightGBM LGBMRegressor (objective=`regression_l1`)
- Внешние признаки (погода): нет
- Исключённые признаки: —
- Свежее окно уровня: 4 нед.
- Число признаков: 25
- Best iteration (early stopping): 1105

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
| LightGBM (+постобработка) | 0.0909 | **0.9091** | 89.64 |
| Блендинг LGBM + медиана(4 нед) | 0.0910 | **0.9090** | 89.80 |

## 4. Топ-20 признаков по важности
| # | Признак | Важность (split) |
|---|---|---|
| 1 | `dayofyear` | 5746 |
| 2 | `hist_rhd_mean` | 3333 |
| 3 | `days_to_holiday` | 2939 |
| 4 | `hour` | 2511 |
| 5 | `day` | 2508 |
| 6 | `hist_rhe_mean` | 2259 |
| 7 | `hist_rh_mean` | 1959 |
| 8 | `hist_rd_med` | 1746 |
| 9 | `days_after_holiday` | 1630 |
| 10 | `route` | 1170 |
| 11 | `hist_rhd_med` | 1131 |
| 12 | `recent_rhd_med` | 914 |
| 13 | `trend_ratio` | 906 |
| 14 | `dow` | 831 |
| 15 | `hist_rd_mean` | 594 |
| 16 | `hist_rh_med` | 536 |
| 17 | `month` | 487 |
| 18 | `hist_r_mean` | 436 |
| 19 | `weekofyear` | 334 |
| 20 | `recent_rh_mean` | 327 |

## 5. Вывод
- WAPE-score модели: **0.9091**, блендинга: **0.9090**.
- Порог baseline (~0.48) и целевой порог 0.85 достигнуты.
