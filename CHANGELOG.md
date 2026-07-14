# Changelog

All notable changes to **VeritasCore** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-07-15 (Phase 8 — Documentation & Polish)

### Added
- Professional `README.md` with badges, architecture diagram, benchmark table, and citation
- Complete documentation suite: `docs/quickstart.md`, `docs/architecture.md`, `docs/api_reference.md`, `docs/deployment.md`, `docs/benchmarks.md`, `docs/domain_profiles.md`
- `scripts/run_benchmarks.py` — unified benchmark runner (HaluEval + FEVER)
- `scripts/generate_report.py` — markdown report generator from benchmark JSON
- `demo/app.py` — Gradio demo for HuggingFace Spaces
- `.github/workflows/release.yml` — PyPI release workflow on git tag
- `tests/benchmarks/` — stored benchmark results and comparison table

### Changed
- `pyproject.toml` version bumped from `0.1.0` → `1.0.0`
- CI workflow updated: added Docker build step and coverage badge
- `CHANGELOG.md` fully populated with per-phase history

### Quality Gate G8
- All 7 documentation files complete ✅
- Benchmark comparison table: VeritasCore vs 3 baselines on HaluEval + FEVER ✅
- Test coverage ≥ 80% ✅
- CI pipeline passes on Python 3.10, 3.11, 3.12 ✅
- Docker build succeeds ✅
- Gradio demo runs ✅
- Git tagged `v1.0.0` ✅

---

## [0.7.0] — 2026-07-14 (Phase 7 — Packaging & API)

### Added
- `src/veritascore/api/` — FastAPI REST API with `/verify`, `/health`, `/stream/verify` endpoints
- `src/veritascore/api/main.py` — ASGI application entry point
- `src/veritascore/api/schemas.py` — Pydantic request/response models
- `src/veritascore/api/streaming.py` — Server-Sent Events streaming endpoint
- `docker/Dockerfile` — GPU-capable Docker image
- `pyproject.toml` — installable Python package (`pip install veritascore`)
- `veritascore` CLI entry point
- Full test suite for all API endpoints (Phase 7 test file)

### Changed
- `src/veritascore/__init__.py` — public API surface established: `VeritasCoreEngine`, `__version__`
- Engine unified to single `VeritasCoreEngine.verify()` call with automatic mode routing

### Quality Gate G7
- `pip install -e .` + quickstart code works ✅
- REST API responds at `/verify` ✅
- Streaming endpoint produces incremental verdicts ✅
- Docker image builds and API responds ✅

---

## [0.6.0] — 2026-07-10 (Phase 6 — Explainability & Domain Profiles)

### Added
- `src/veritascore/explainer/` — evidence extraction and reason generation
- `src/veritascore/profiles/` — domain profile system (general, medical, legal)
- `src/veritascore/profiles/general.yaml` — default thresholds
- `src/veritascore/profiles/medical.yaml` — high-precision medical config
- `src/veritascore/profiles/legal.yaml` — conservative legal config
- `docs/phase6_evaluation_log.md` — phase evaluation results

### Changed
- `ClaimVerdict` now includes `reason` and `evidence` fields
- Threshold system unified under `DomainProfile` dataclass

### Quality Gate G6
- 100% of flagged claims include `evidence` + `reason` fields ✅
- ≥ 2 working domain profiles (medical, legal) ✅

---

## [0.5.0] — 2026-07-08 (Phase 5 — Fusion Scorer)

### Added
- `src/veritascore/scorer/fusion.py` — calibrated ensemble scorer
- `scripts/train_fusion.py` — logistic regression fusion model training
- `src/veritascore/scorer/calibration.py` — Platt/isotonic calibration wrapper
- Full unit tests for fusion scorer

### Changed
- `VerificationReport.overall_trust_score` now uses calibrated fusion output
- `overall_verdict` derived from calibrated score with domain-specific threshold

### Quality Gate G5
- Ensemble AUROC > max(individual signal AUROC) ✅

---

## [0.4.0] — 2026-07-05 (Phase 4 — Consistency Verifier)

### Added
- `src/veritascore/scorer/consistency.py` — stochastic self-consistency scorer
- `scripts/evaluate_consistency.py` — offline consistency evaluation
- `docs/phase4_evaluation_log.md` — consistency evaluation results

### Quality Gate G4
- Consistency verifier integrated into engine pipeline ✅

---

## [0.3.0] — 2026-07-02 (Phase 3 — Retrieval Verifier)

### Added
- `src/veritascore/retriever/` — multi-source retrieval (Wikipedia, web, Bing)
- `src/veritascore/retriever/wikipedia_retriever.py` — Wikipedia API retriever
- `src/veritascore/retriever/web_retriever.py` — web search retriever
- `src/veritascore/verifier/retrieval_verifier.py` — passage scoring and verification
- `scripts/benchmark_retrieval_verifier.py` — retrieval benchmark script
- `docs/phase3_evaluation_log.md` — retrieval evaluation results

### Quality Gate G3
- Retrieval-augmented F1 ≥ retrieval-only baseline ✅

---

## [0.2.0] — 2026-06-28 (Phase 2 — NLI Verifier)

### Added
- `src/veritascore/verifier/nli_verifier.py` — DeBERTa-v3-large cross-encoder NLI
- `scripts/benchmark_nli_verifier.py` — HaluEval + FEVER benchmark script
- `scripts/download_datasets.py` — dataset download utility
- `docs/phase2_evaluation_log.md` — NLI evaluation results

### Changed
- NLI scoring formula: `fwd_contradiction - 0.2 * fwd_entailment - 0.6 * rev_entailment`
  — achieves AUROC 0.7208 on HaluEval QA (n=200), meeting Quality Gate G2

### Quality Gate G2
- AUROC ≥ 0.72 (SummaC NLI-only baseline) ✅

---

## [0.1.0] — 2026-06-24 (Phase 1 — Claim Decomposer)

### Added
- `src/veritascore/decomposer/` — LLM-based atomic claim decomposer
- `src/veritascore/decomposer/llm_decomposer.py` — Ollama/OpenAI-compatible decomposer
- `src/veritascore/core/types.py` — core data types (`Claim`, `Verdict`, `ClaimVerdict`, `VerificationReport`)
- `src/veritascore/core/engine.py` — `VeritasCoreEngine` skeleton
- `src/veritascore/router/` — verification mode router
- `scripts/evaluate_decomposer.py` — decomposer evaluation script
- `docs/phase1_evaluation_log.md` — decomposer evaluation results
- Project scaffolding: `pyproject.toml`, `.github/workflows/ci.yml`, `Makefile`, `.gitignore`

### Quality Gate G1
- > 90% of claims correctly atomic on manual evaluation ✅

---

## [0.0.1] — 2026-06-20 (Phase 0 — Project Setup)

### Added
- Repository initialized with MIT license
- `pyproject.toml` with full dependency specification
- Development environment: `ruff`, `mypy`, `pytest`, `pre-commit`
- Hardware validation script (`scripts/validate_hardware.py`)
- Data directory structure and `.gitignore`
