# Makefile — воспроизводимость ML-пайплайна
# Windows: запускать через `make` (GNU Make) или напрямую команды python.

PYTHON ?= python

.PHONY: help install data eda baseline external train train-v1 train-v2 predict validate test all clean

help:
	@echo "Доступные цели:"
	@echo "  install    - установить зависимости (pip install -r requirements.txt)"
	@echo "  data       - подготовить data/processed/boardings.parquet"
	@echo "  external   - обновить внешние данные (погода, праздники)"
	@echo "  baseline   - посчитать baseline и записать reports/baseline.md"
	@echo "  train      - обучить v1 и v2, записать отчёты"
	@echo "  predict     - финальный прогноз -> submissions/*.csv"
	@echo "  validate   - проверить формат submissions/submission.csv"
	@echo "  test       - запустить pytest"
	@echo "  all        - data + train + predict + validate"

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data

external:
	$(PYTHON) -m src.external.weather
	$(PYTHON) -m src.external.holidays

baseline:
	$(PYTHON) -m src.baseline

train-v1:
	$(PYTHON) -m src.train --out models/lgbm_v1.txt --report reports/model_v1.md

train-v2:
	$(PYTHON) -m src.train --exclude "month,dayofyear,weekofyear,day" --weather \
		--recent-weeks 8 --out models/lgbm_v2.txt --report reports/model_v2.md \
		--importance-csv reports/feature_importance.csv

train: train-v1 train-v2

predict:
	$(PYTHON) -m src.predict

validate:
	$(PYTHON) -m src.validate_submission

test:
	$(PYTHON) -m pytest tests -q

all: data train predict validate

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['src/__pycache__','src/external/__pycache__','tests/__pycache__','.pytest_cache']]"
