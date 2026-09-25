# CONVENTIONS.md — код-стайл и структура

## Язык

- Python 3.10+.
- Комментарии и docstrings — на русском или английском, единообразно.
- Имена переменных/функций — английский.

## Структура репозитория
.
├── ai/ # эти файлы для агента
├── data/
│ ├── raw/ # исходные labels (копия из архива)
│ ├── external/ # погода, праздники
│ └── processed/ # parquet, промежуточные
├── labels/ # как выдано организаторами
├── spravochniki/ # как выдано организаторами
├── models/ # сохранённые модели
├── notebooks/ # EDA
├── reports/ # markdown-отчёты
├── src/
│ ├── data.py
│ ├── features.py
│ ├── train.py
│ ├── predict.py
│ ├── validate_submission.py
│ ├── main.py
│ └── external/
│ ├── weather.py
│ └── holidays.py
├── submissions/
│ ├── submission.csv
│ └── forecast.csv
├── tests/
├── requirements.txt
├── Makefile
└── README_ML.md

text

## Стиль кода

- PEP 8.
- Type hints обязательны для публичных функций.
- `pathlib.Path` вместо `os.path`.
- Логирование через `logging`, не `print` (кроме отладки).
- Никаких «магических» чисел без констант.
- Функции — маленькие, тестируемые.

## Зависимости

`requirements.txt`:
pandas
numpy
scikit-learn
lightgbm
catboost
openpyxl
requests
pyarrow
tqdm
matplotlib
seaborn

text

## Конфиги

- Гиперпараметры и пути — в `config.yaml` или `src/config.py`.
- Seed фиксирован: `RANDOM_SEED = 42`.

## Тесты

Минимум:

- `test_data.py` — загрузка, сетка, отсутствие route=5.
- `test_submission.py` — формат сабмита.

## Git

- Ветки: `main`, `ml/<task>`, `web/<task>`.
- Коммиты осмысленные.
- Хостинг: gitverse (российский, по требованию организаторов).

## Логи

- `logs/` в `.gitignore`.
- Уровни: INFO по умолчанию, DEBUG по флагу.

## Секреты

- Не коммитить токены/пароли.
- `.env` + `.env.example`.