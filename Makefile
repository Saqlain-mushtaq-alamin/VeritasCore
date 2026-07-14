.PHONY: install dev test lint format type-check validate-hw download-models download-data clean help benchmark demo generate-report tag-release

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

# ── Quality ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v -m "not integration" --cov=veritascore --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v -m integration

test-all:
	pytest tests/ -v --cov=veritascore --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

type-check:
	mypy src/veritascore/ --ignore-missing-imports

check: lint type-check test

# ── Setup Scripts ─────────────────────────────────────────────────────────────

validate-hw:
	python scripts/validate_hardware.py

download-models:
	python scripts/download_models.py

download-data:
	python scripts/download_datasets.py

benchmark-decomposer:
	python scripts/evaluate_decomposer.py --decomposer both --verbose

benchmark-nli:
	python scripts/benchmark_nli_verifier.py --dataset both --n 200

benchmark-retrieval:
	python scripts/benchmark_retrieval_verifier.py --dataset halueval --n 50

benchmark-consistency:
	python scripts/evaluate_consistency.py --verbose

train-fusion:
	python scripts/train_fusion.py --n 300 --model logistic_regression

serve:
	uvicorn veritascore.api.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn veritascore.api.app:app --host 0.0.0.0 --port 8000 --workers 2

docker-build:
	docker build -f docker/Dockerfile -t veritascore:latest .

docker-run:
	docker run -p 8000:8000 --env-file .env veritascore:latest

clear-search-cache:
	python -c "from veritascore.retriever.cache import SearchCache; n = SearchCache().clear(); print(f'Cleared {n} cached search results')"

setup: dev validate-hw download-models download-data
	@echo "✓ Full setup complete"

# ── Phase 8: Benchmarks, Demo, Release ────────────────────────────────────────

benchmark:
	python scripts/run_benchmarks.py --datasets halueval fever --n 200
	python scripts/generate_report.py

generate-report:
	python scripts/generate_report.py

demo:
	pip install -e ".[demo]" -q
	python demo/app.py

tag-release:
	@echo "Tagging v1.0.0 ..."
	git tag -a v1.0.0 -m "Release v1.0.0 — Phase 8 complete: Documentation & Polish"
	@echo "Push with: git push origin v1.0.0"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "VeritasCore — Development Commands"
	@echo ""
	@echo "  make install         Install package"
	@echo "  make dev             Install with dev dependencies"
	@echo "  make test            Run all tests with coverage"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Format code with ruff"
	@echo "  make type-check      Run mypy type checker"
	@echo "  make check           lint + type-check + test"
	@echo "  make validate-hw     Check hardware requirements"
	@echo "  make download-models Download all ML models"
	@echo "  make download-data   Download benchmark datasets"
	@echo "  make setup           Full first-time setup"
	@echo "  make clean           Remove cache and build artifacts"
	@echo ""
	@echo "Phase 8:"
	@echo "  make benchmark        Run HaluEval + FEVER benchmarks"
	@echo "  make generate-report  Generate comparison table from results"
	@echo "  make demo             Launch Gradio demo locally"
	@echo "  make tag-release      Tag git commit as v1.0.0"
