PY ?= python3
EVAL ?= data/eval.csv
OUT ?= predictions.csv
export PYTHONPATH := src

.PHONY: setup install resources index audit splits train final-test predict test bench errors drift report all kaggle tiny-models

install:
	$(PY) -m pip install -r requirements.txt

resources:
	$(PY) scripts/setup_resources.py

index:
	$(PY) scripts/build_index.py --config configs/evaluation.yaml

setup: install resources index

audit:
	$(PY) scripts/audit.py

splits:
	$(PY) scripts/make_splits.py

train: audit splits
	$(PY) scripts/train.py --config configs/evaluation.yaml --tag metrics
	$(PY) scripts/evaluate.py --render

final-test: audit splits
	$(PY) scripts/train.py --config configs/evaluation.yaml --tag metrics --final-test
	$(PY) scripts/evaluate.py --render

errors:
	$(PY) scripts/error_analysis.py --tag val --run reports/private/val_run.pkl
	$(PY) scripts/error_analysis.py --tag code_disjoint --run reports/private/code_disjoint_run.pkl
	$(PY) scripts/error_analysis.py --tag final_test --run reports/private/final_test_run.pkl

kaggle:
	bash scripts/kaggle_run.sh

tiny-models:
	$(PY) scripts/make_tiny_test_models.py

predict:
	$(PY) -m termmap.predict --input $(EVAL) --output $(OUT) --config configs/evaluation.yaml

test:
	$(PY) -m pytest -q

bench:
	$(PY) scripts/benchmark_speed.py

drift:
	$(PY) scripts/drift_report.py --icd $(NEW_ICD) --snomed $(NEW_SNOMED)

report:
	$(PY) scripts/make_experiment_log.py
	$(PY) scripts/make_summary.py
	$(PY) scripts/make_pdf.py

all: setup train errors test bench report
