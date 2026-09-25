# EXTERNAL_DATA.md — внешние данные

Q&A официально разрешает использовать внешние данные, опубликованные
после 31.10.2025, в том числе фактическую погоду и новости Дептранса.

## Погода (Open-Meteo Archive)

Москва: `lat=55.7558`, `lon=37.6173`.

URL:
https://archive-api.open-meteo.com/v1/archive
?latitude=55.7558
&longitude=37.6173
&start_date=2025-11-01
&end_date=2025-12-31
&hourly=temperature_2m,precipitation,snowfall,snow_depth,wind_speed_10m,cloud_cover,relative_humidity_2m
&timezone=Europe/Moscow

text

Сохранять в `data/external/weather_2025_11_12.csv`.

Колонки после обработки:

- `date`, `hour`, `temperature_2m`, `precipitation`, `snowfall`,
  `snow_depth`, `wind_speed_10m`, `cloud_cover`, `relative_humidity_2m`.

## Праздники РФ

- 4 ноября — День народного единства.
- 31 декабря — короткий/предпраздничный день.
- Опционально: школьные каникулы (конец октября — начало ноября,
  конец декабря).

Сохранять в `data/external/holidays_2025.csv`.

## Расписания транспорта

Из Q&A: расписания из открытых источников можно использовать.
Это может помочь для маршрутов с ограничениями движения.

## Новости Дептранса

Можно искать упоминания ремонтов и изменений маршрутов в ноябре–декабре
2025. Если найдёшь — добавить фичи «ремонт на маршруте».

## Кэширование

Все внешние данные кэшировать в `data/external/`. Не бить API каждый
раз при перезапуске.