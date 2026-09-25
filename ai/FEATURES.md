# FEATURES.md — признаки

## Базовые календарные

- `route` — категория.
- `hour` — 0…23.
- `dayofweek` — 0…6.
- `is_weekend` — 0/1.
- `month` — 1…12.
- `day` — 1…31.
- `weekofyear` — номер недели.
- `dayofyear` — номер дня в году.
- `is_holiday` — 0/1.
- `is_preholiday` — 0/1 (например, 31 декабря, 3 ноября).
- `days_to_holiday` — сколько дней до ближайшего праздника.
- `days_after_holiday`.

## Исторические агрегаты

Важно: на валидации считать **только по train-части**, чтобы не было
утечки.

- `route + hour` — mean, median.
- `route + hour + dow` — mean, median.
- `route + dow` — mean, median.
- `route + hour + is_weekend` — mean.
- `route` — mean, median, std.
- Лаги: `boardings` за 1, 7, 14, 28 дней назад для того же `route + hour`.
- Скользящие средние: 7, 14, 28 дней для `route + hour`.

## Внешние

Погода (Open-Meteo Archive) по Москве за ноябрь–декабрь 2025:

- `temperature_2m` — температура.
- `precipitation` — осадки.
- `snowfall` — снег.
- `snow_depth` — снежный покров.
- `wind_speed_10m` — ветер.
- `cloud_cover` — облачность.
- `relative_humidity_2m` — влажность.

Джойн по `date + hour`.

## Признаки аномалий

- Флаг «был ли провал в предыдущие 7 дней» для `route`.
- Флаг «технологический час» (2–4).
- Флаг «первый/последний день месяца».
- Флаг «первая неделя после каникул».

## Что НЕ использовать

- `input_date_time` из сырых CSV.
- `pass_route` — зашумлено.
- `place_id` как остановку — это депо.
- Любые поля, которые могут содержать утечку из будущего.