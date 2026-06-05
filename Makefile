.PHONY: help install test test-unit test-func test-all lint clean run-dashboard run-sequential run-sim run-baseline generate-consensus train-simgnn run-evals typecheck

PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
UV     := uv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies via uv
	$(UV) sync
	$(UV) pip install torch-geometric

test-unit:  ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-func:  ## Run functional/integration tests
	$(PYTHON) -m pytest tests/functional/ -v --tb=short

test-all:  ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test: test-unit  ## Run unit tests (default)

lint:  ## Run flake8 and isort checks
	$(PYTHON) -m flake8 client/ server/ adversary/ eval/ --max-line-length=120 --ignore=E203,W503
	$(PYTHON) -m isort --check-only --profile black client/ server/ adversary/

typecheck:  ## Run mypy static type checking
	$(PYTHON) -m mypy client/ --ignore-missing-imports

clean:  ## Remove caches, pyc files, and saved models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf saved_models/

run-dashboard:  ## Launch the Streamlit dashboard
	$(UV) run streamlit run app.py

run-sequential:  ## Run sequential (non-Ray) FL simulation
	$(PYTHON) run_sim_sequential.py

run-sim:  ## Run full Ray-based FL simulation
	$(PYTHON) federated_sim.py

run-baseline:  ## Run the FedAvg baseline simulation
	$(PYTHON) baseline_fedavg_sim.py

generate-consensus:  ## Build the honest consensus DAG + pre-train SimGNN
	$(PYTHON) server/generate_consensus.py

train-simgnn:  ## Fine-tune SimGNN on the consensus graph
	$(PYTHON) server/train_simgnn.py

run-evals:  ## Run the paper evaluation suite
	$(PYTHON) -m eval.run_all_evals

run-evals-fast:  ## Run fast evaluation suite (skip slow)
	$(PYTHON) -m eval.run_all_evals --skip-slow

validate-config:  ## Validate params.yaml configuration
	$(PYTHON) -c "from server.config_validator import validate_config; validate_config(); print('Config OK')"

full-pipeline: generate-consensus train-simgnn run-sequential run-evals-fast  ## Run the full paper pipeline
