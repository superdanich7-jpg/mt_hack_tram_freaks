# api_spec.md — контракт ML → backend

Источник данных для сервиса — `submissions/forecast.csv`
(`route;date;hour;prediction`, UTF-8, разделитель `;`).
Backend читает только этот файл (или загруженный в память DataFrame) —
сырые CSV разбирать нельзя.

Общие правила:

- Формат ответа: `application/json; charset=utf-8`.
- `date` в API — `YYYY-MM-DD`, `hour` — целое `0…23`.
- `route` — только из `[1,5,7,11,12,17,25,26,28,50]`.
- `prediction` — целое `≥ 0`. Для `route=5` всегда `0`.
- Горизонт прогноза: `2025-11-01` … `2025-12-31` (61 день).
- Ошибки: `400` — неверные параметры, `404` — нет данных.
  Тело ошибки: `{"error": "<код>", "message": "<человекочитаемо>"}`.

---

## 1. `GET /routes`

Список маршрутов с агрегатами за весь прогнозный период.

**Ответ 200**

```json
{
  "routes": [
    {"route": 1,  "days": 61, "total_prediction": 1058756, "avg_daily": 17356.7},
    {"route": 5,  "days": 61, "total_prediction": 0,       "avg_daily": 0.0},
    {"route": 17, "days": 61, "total_prediction": 2900212, "avg_daily": 47544.5},
    {"route": 50, "days": 61, "total_prediction": 1246004, "avg_daily": 20426.3}
  ]
}
```

| Поле | Тип | Описание |
|---|---|---|
| `route` | int | номер маршрута |
| `days` | int | число дней с прогнозом |
| `total_prediction` | int | сумма посадок за период |
| `avg_daily` | float | среднее в день |

---

## 2. `GET /forecast?route=1&date=2025-11-01`

Прогноз по одному маршруту на один день — 24 значения.

**Параметры**

| Имя | Тип | Обяз. | Описание |
|---|---|---|---|
| `route` | int | да | маршрут |
| `date` | str | да | `YYYY-MM-DD` |

**Ответ 200**

```json
{
  "route": 1,
  "date": "2025-11-01",
  "dow": 5,
  "is_holiday": false,
  "total": 10533,
  "peak_hour": 13,
  "peak_value": 1045,
  "hours": [
    {"hour": 0, "prediction": 3},
    {"hour": 1, "prediction": 0}
  ]
}
```

(Значения соответствуют реальному `submissions/forecast.csv`.)

**Ошибки**: `400` при `route` вне списка или неверной дате; `404` если пары нет в прогнозе.

---

## 3. `GET /forecast/week?route=1&start=2025-11-01`

Прогноз на 7 дней вперёд от `start` (включительно).

**Параметры**

| Имя | Тип | Обяз. | Описание |
|---|---|---|---|
| `route` | int | да | маршрут |
| `start` | str | да | первый день окна `YYYY-MM-DD` |

**Ответ 200**

```json
{
  "route": 1,
  "start": "2025-11-01",
  "end": "2025-11-07",
  "days": [
    {"date": "2025-11-01", "dow": 5, "is_holiday": false, "total": 10533,
     "hours": [{"hour": 0, "prediction": 3}]}
  ]
}
```

Если окно выходит за `2025-12-31`, лишние дни отбрасываются (`400`, если
`start` позже `2025-12-25`).

---

## 4. `GET /forecast/month?route=1&month=2025-11`

Прогноз на календарный месяц.

**Параметры**

| Имя | Тип | Обяз. | Описание |
|---|---|---|---|
| `route` | int | да | маршрут |
| `month` | str | да | `YYYY-MM` (только `2025-11` или `2025-12`) |

**Ответ 200**

```json
{
  "route": 1,
  "month": "2025-11",
  "days": 30,
  "total": 497508,
  "weekday_avg": 80464.0,
  "weekend_avg": 47594.0,
  "days_detail": [
    {"date": "2025-11-01", "dow": 5, "is_holiday": false, "total": 10533}
  ]
}
```

---

## 5. `GET /anomalies`

Маршруты и дни, где прогноз сильно отклоняется от собственной нормы
(например, `|prediction_day − median_same_dow| / median_same_dow > 0.35`
или `is_holiday = true`). Полезно для подсветки в UI.

**Ответ 200**

```json
{
  "threshold": 0.35,
  "items": [
    {"route": 17, "date": "2025-11-03", "type": "holiday_effect",
     "dow": 0, "total": 38211, "expected": 51230, "deviation": -0.254},
    {"route": 50, "date": "2025-12-31", "type": "spike",
     "dow": 2, "total": 21004, "expected": 15400, "deviation": 0.364}
  ]
}
```

(пример структуры ответа)

| Поле | Тип | Описание |
|---|---|---|
| `type` | str | `holiday_effect` / `spike` / `drop` |
| `total` | int | сумма прогноза за день (все часы маршрута) |
| `expected` | float | норма для того же дня недели в окне прогноза |
| `deviation` | float | относительное отклонение от нормы `(total − expected)/expected` |

---

## 6. `POST /predict` (опционально, онлайн-инференс)

Для запуска модели «на лету», если сервис не использует готовый `forecast.csv`.

**Запрос**

```json
{"route": 1, "date": "2025-11-15", "hour": 18,
 "weather": {"temperature_2m": -3.5, "precipitation": 0.2, "snowfall": 0.0,
             "snow_depth": 1.0, "wind_speed_10m": 4.1,
             "cloud_cover": 90.0, "relative_humidity_2m": 85.0}}
```

**Ответ 200**

```json
{"route": 1, "date": "2025-11-15", "hour": 18, "prediction": 2140}
```

Поля `weather` необязательны: при отсутствии берётся кэш `data/external/weather.csv`.
При `route = 5` возвращается `{"route": 5, "prediction": 0}`.

---

## 7. Коды и лимиты

| Код | Когда |
|---|---|
| `200` | успех |
| `400` | неверные параметры (route/date/hour вне допустимых значений) |
| `404` | данных нет в прогнозе |
| `500` | внутренняя ошибка чтения `forecast.csv` |

Ответы кэшируются (`forecast.csv` статичен) — рекомендуется заголовок
`Cache-Control: public, max-age=3600`.

Аутентификация — базовая (требование организаторов), логин/пароль в `.env`,
не в репозитории.
