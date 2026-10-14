# Reproducing VeritasCore Results

This document provides exact, step-by-step instructions to reproduce all results reported in the VeritasCore paper.

## Environment Setup

```bash
git clone https://github.com/Saqlain-mushtaq-alamin/VeritasCore.git
cd VeritasCore

# Create a virtual environment
python -m venv .venv
# Activate it:
source .venv/bin/activate        # Linux / macOS
# or
.venv\Scripts\activate           # Windows

# Install exact locked versions (for exact reproducibility of reported results)
pip install -r requirements-lock.txt

# Install the package in editable mode (with dev extras for testing)
pip install -e ".[dev]"
```

## Configure API Keys (optional — required for retrieval verifier)

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
# Edit .env and add your Brave Search or Tavily API key
```

> **Note:** API keys are only required for the Retrieval Verifier (ungrounded mode). NLI-only benchmarks run fully locally without any API key.

## Download Models

```bash
python scripts/download_models.py
```

This will download and cache:
- `cross-encoder/nli-deberta-v3-base` (rev `6c749ce`)
- `sentence-transformers/all-MiniLM-L6-v2` (rev `1110a24`)
- `microsoft/Phi-3-mini-4k-instruct` (rev `f39ac1d`)

## Download Datasets

```bash
python scripts/download_datasets.py --only halueval fever
```

## Run Benchmarks

### 1. Main Benchmark: VeritasCore NLI Evaluation (Phase R3)

Run the unified evaluation script on the disjoint held-out test split with bootstrapped 95% confidence intervals and paired statistical significance tests:

```bash
python scripts/run_benchmarks_v2.py \
    --datasets halueval fever \
    --n 1500 \
    --grid-n 500 \
    --split eval \
    --seed 42 \
    --bootstrap-ci \
    --n-bootstrap 2000 \
    --significance-test \
    --output tests/benchmarks/results/
```

### 2. Paired SummaC ZS Oracle Baseline (Phase R4)

Run the reimplemented SummaC ZS baseline on the exact same held-out split and sample order:

```bash
python scripts/baseline_summac.py \
    --dataset both \
    --n 1500 \
    --grid-n 500 \
    --split eval \
    --seed 42 \
    --bootstrap-ci \
    --n-bootstrap 2000 \
    --output tests/benchmarks/results/
```

### 3. Reliability Diagrams & Error Analysis (Phase R7)

Fit Platt scaling on the grid split and evaluate calibration (ECE, Brier score) on the held-out evaluation split:

```bash
python scripts/generate_calibration_plot.py \
    --datasets halueval fever \
    --n 500 \
    --seed 42 \
    --output results/calibration/
```

### 4. Ablation Study (Phase R5)

Measure incremental signal contributions across components:

```bash
python scripts/run_ablation.py \
    --dataset halueval \
    --n 500 \
    --seed 42 \
    --output results/ablation/
```

### 5. Run the Test Suite

```bash
pytest tests/ -v
```

### 6. Static Analysis

```bash
ruff check src/
mypy src/ --ignore-missing-imports
```

## Hardware Used for Reported Results

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) |
| CPU | Intel Core i7-12th Gen |
| RAM | 16 GB DDR5 |
| OS | Windows 11 / Ubuntu 22.04 |
| Python | 3.11.x |
| CUDA | 12.x |
| PyTorch | See `requirements-lock.txt` |

## Model Revisions Used

All model revisions are pinned in `configs/models/model_registry.yaml` and loaded via `ModelConfig.nli_revision`, `embedding_revision`, and `decomposer_revision`. This guarantees identical weights are loaded across machines:

| Model | Pinned Revision |
|-------|----------------|
| `cross-encoder/nli-deberta-v3-base` | `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |

## Evaluation Methodology & Protocol

- **Train/Test Split Separation**: HaluEval QA is strictly partitioned into non-overlapping splits: a 500-sample grid search set (`split="grid"`) used for hyperparameter selection / Platt scaling calibration, and an independent held-out evaluation set (`split="eval"`, n=1500) for all reported metrics.
- **Strictly Paired Baselines**: SummaC ZS baseline uses the identical held-out eval samples, random seed (42), and label mapping (`split="eval"`). Both VeritasCore and SummaC store raw `y_true` and `y_score` arrays.
- **Statistical Significance**: DeLong's test and paired bootstrap tests (2,000 resamples) compute two-sided p-values and 95% confidence intervals on AUROC differences between paired models.
- **TruthfulQA Protocol**: Evaluated strictly as an exploratory negative probe. Since questions cannot semantically entail answers without external reference knowledge, ungrounded NLI evaluates prior bias rather than factual entailment.
- **FEVER Protocol**: Evaluated on gold/oracle validation evidence with non-NEI samples collected systematically. See `docs/benchmarks.md` for detailed discussion.
