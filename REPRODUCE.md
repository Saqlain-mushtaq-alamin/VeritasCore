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

### NLI-only benchmark (recommended starting point)

```bash
python scripts/run_benchmarks.py --datasets halueval fever --n 1000
```

### Full pipeline benchmark (requires API keys in .env)

```bash
python scripts/run_benchmarks.py --datasets halueval fever --n 1000 --mode all
```

### Run the test suite

```bash
pytest tests/ -v
```

### Static analysis

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

All model revisions are pinned in `configs/models/model_registry.yaml`. The exact commit hashes guarantee that the same model weights are loaded regardless of future upstream changes.

| Model | Pinned Revision |
|-------|----------------|
| `cross-encoder/nli-deberta-v3-base` | `6c749ce3425cd33b46d187e45b92bbf96ee12ec7` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `microsoft/Phi-3-mini-4k-instruct` | `f39ac1d28e925b323eae81227eaba4464caced4e` |
| `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` |

## Notes on Reported Results

- **HaluEval AUROC 0.721**: NLI-only mode, n=200 samples (held-out split). Grid search was performed on a separate 500-sample split.
- **FEVER AUROC 0.72**: NLI-only mode, oracle (gold) evidence provided — not end-to-end retrieval. See `docs/benchmarks.md` for full methodology details and caveats.
