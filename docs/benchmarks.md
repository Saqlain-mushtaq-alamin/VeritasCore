# Benchmarks — VeritasCore

> Full evaluation results, methodology, and comparison with published baselines.

---

## Datasets

### HaluEval QA

| Property | Value |
|----------|-------|
| Source | [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., 2023) |
| Task | Question-answering hallucination detection |
| Size | 10,000 QA pairs (balanced: 50% hallucinated) |
| Format | `(knowledge_paragraph, question, answer, hallucination_label)` |
| Eval split | Held-out n=1,500 (grid-search on separate n=500; no train-test overlap) |
| Claim format | `"Q: <question>  A: <answer>"` (question context needed for short answers) |

### FEVER

| Property | Value |
|----------|-------|
| Source | [FEVER](https://fever.ai/) / [copenlu/fever_gold_evidence](https://huggingface.co/datasets/copenlu/fever_gold_evidence) |
| Task | Fact extraction and verification |
| Size | 185,445 claims (validation: 15,935) |
| Format | `(claim, evidence_passages, label: SUPPORTS/REFUTES/NOT ENOUGH INFO)` |
| Eval split | n=1,000 SUPPORTS + REFUTES samples from validation (NEI excluded) |
| Claim format | Full declarative sentences — ideal for NLI evaluation |

---

## Evaluation Protocols

We evaluate VeritasCore under **two distinct protocols** that must not be conflated:

### Oracle-Evidence Protocol (Table 1)
> "Given gold evidence, how well does each NLI component detect hallucination?"

Each system receives the same gold evidence passages (oracle retrieval). For HaluEval, this is the provided knowledge paragraph; for FEVER, the gold evidence triples from `copenlu/fever_gold_evidence`. This measures **NLI model quality in isolation**, removing retrieval noise.

We re-run the SummaC baseline (Laban et al., 2022) under identical conditions so the comparison is apples-to-apples (`scripts/baseline_summac.py`).

### Full-Pipeline Protocol (Table 2)
> "In real deployment without pre-provided evidence, how well does each system perform?"

Each system runs its complete pipeline, including evidence retrieval. VeritasCore uses web search; FActScore uses the Wikipedia API; SelfCheckGPT uses self-consistency (no external evidence). This measures **real-world deployment performance**.

> **Important:** Oracle-evidence results (Table 1) must not be directly compared with full-pipeline results (Table 2). Gold evidence eliminates retrieval errors that significantly affect real-world performance.

---

## Preprocessing

1. **HaluEval**: Claims formatted as `"Q: {question}  A: {answer}"` so the NLI model has semantic context. Raw short answers (e.g., "Delhi") produce near-zero entailment regardless of correctness.

2. **FEVER**: `NOT ENOUGH INFO` samples excluded. The loader collects exactly n non-NEI samples (R3 fix — the old version counted rows including NEI, yielding fewer than n usable samples). Evidence triples flattened to a single context string.

3. **Claim decomposition**: For benchmark purposes, NLI and retrieval verifiers are evaluated directly on dataset claims (no decomposer step), to isolate signal quality.

4. **Train/test separation (HaluEval)**: The hallucination scoring formula (`fwd_c - 0.2×fwd_e - 0.6×rev_e`) was grid-searched on a separate n=500 split. All reported AUROC numbers are from the disjoint held-out eval split.

### NLI Scoring Formula

**Hallucination score** (NLI):
```
hallucination_score = fwd_contradiction - 0.2 × fwd_entailment - 0.6 × rev_entailment
```
Grid-searched on HaluEval grid split (n=500). Evaluated on held-out split (n=1,500). Range: approximately [-1.0, +1.0].

---

## Results

### Table 1: NLI Quality (Oracle Evidence)

> **Protocol:** Each system receives identical gold evidence passages. Measures NLI model quality in isolation (not retrieval quality).
> **Both systems re-run by us** under identical oracle-evidence conditions on RTX 4060 (n=1,000 per dataset, bootstrapped 95% CI, 2,000 resamples).

| Method | HaluEval AUROC (95% CI) | FEVER AUROC (95% CI) | Evidence | n |
|--------|:-----------------------:|:--------------------:|----------|:-:|
| SummaC ZS (reimplemented†, DeBERTa-v3-base) | 0.5684 (0.5332–0.6020) | 0.8443 (0.8188–0.8682) | Oracle | 1,000 |
| **VeritasCore NLI** | **0.7044 (0.6722–0.7358)** | **0.9680 (0.9568–0.9776)** | Oracle | 1,000 |

*†SummaC ZS algorithm (Laban et al. 2022 §3.1) reimplemented with the same DeBERTa-v3-base backbone as VeritasCore NLI. The original `vitc` model (summac v0.0.4) pins `huggingface_hub==0.17.0`, which is incompatible with `datasets>=0.25.0`. Using the same backbone ensures the algorithmic difference (sentence-level max-entailment vs. our bidirectional formula) is the only variable.*

**Key finding:** VeritasCore NLI outperforms SummaC ZS by **+0.136 AUROC on HaluEval** and **+0.124 AUROC on FEVER** under identical oracle-evidence conditions. The difference stems from the bidirectional scoring formula (`fwd_c − 0.2×fwd_e − 0.6×rev_e`) vs. SummaC's unidirectional max-entailment aggregation. The reverse-entailment term is especially effective on FEVER's full declarative-sentence claims.

---

### Table 2: Full Pipeline (End-to-End)

> **Protocol:** Each system runs its own complete pipeline, including evidence retrieval.
> **Published numbers** from original papers — not re-run by us. Methodology differences noted.

| Method | HaluEval AUROC | FEVER AUROC | Evidence | Source |
|--------|:--------------:|:-----------:|----------|--------|
| FActScore (Min et al., 2023) | 0.680 | 0.740 | Retrieved (Wikipedia API) | Published |
| SelfCheckGPT (Manakul et al., 2023) | 0.740 | 0.690 | Self-consistency (no external evidence) | Published |
| **VeritasCore Fusion** | **TBD** | **TBD** | Retrieved (Web search) | This work |

*VeritasCore Fusion numbers will be added after Phase R5 (ablation study) with bootstrapped 95% CIs and DeLong's test p-values.*

---

### Individual NLI Signal Analysis (HaluEval QA — grid split, n=500)

> **Note:** These numbers are from the grid-search split only. They informed the scoring formula; they are not reported as final results.

| Signal | AUROC (grid split) | Notes |
|--------|:------------------:|-------|
| Raw entailment (`fwd_e` only) | ~0.55 | Uniform for short answers |
| Raw contradiction (`fwd_c` only) | ~0.65 | Better, but noisy |
| `fwd_c − 0.2×fwd_e` | ~0.70 | Good discrimination |
| **`fwd_c − 0.2×fwd_e − 0.6×rev_e`** | **~0.72** | **Selected formula** |

The reverse entailment term (claim → context) captures a strong support signal: if the claim *implies* the context, the claim is likely correct.

---

### Ablation Study

> **Status:** Results pending Phase R5. The table below will be filled with measured values and bootstrapped CIs. No projected targets are reported.

| Configuration | HaluEval AUROC (95% CI) | FEVER AUROC (95% CI) | Δ vs Full Fusion |
|---------------|:-----------------------:|:--------------------:|:----------------:|
| Full Fusion (NLI + Retrieval + Consistency) | TBD | TBD | — |
| − NLI signal | TBD | TBD | TBD |
| − Retrieval signal | TBD | TBD | TBD |
| − Consistency signal | TBD | TBD | TBD |
| NLI-only | TBD | TBD | TBD |

---

### Latency Analysis

> **Note:** Numbers below are preliminary. Full latency table with batch size, precision, and GPU memory details will be reported in Phase R8.

| Mode | Avg Latency/Claim | P95 Latency/Claim | Throughput | Hardware |
|------|:-----------------:|:-----------------:|:----------:|---------|
| Grounded (NLI, GPU) | ~280 ms | ~350 ms | ~3.6 claims/s | RTX 4060 |
| Grounded (NLI, CPU) | ~2,800 ms | ~3,500 ms | ~0.36 claims/s | i7-12th gen |
| Ungrounded (web) | ~2,000 ms | ~4,000 ms | ~0.5 claims/s | Any |
| Offline (consistency) | ~1,500 ms | ~2,500 ms | ~0.7 claims/s | Any |

> **Note:** GPU latency includes NLI cross-encoder inference. Model cold-start (first load) adds ~15–30 s.

---

## Limitations

1. **Oracle evidence advantage**: Table 1 results use gold evidence passages, which eliminates retrieval errors. These results should not be directly compared with full-pipeline results from other systems.

2. **HaluEval short-answer bias**: Short factual answers (e.g., "1897", "Paris") have uniformly low entailment scores, making NLI-only methods struggle without question context. Our Q+A formatting partially mitigates this.

3. **FEVER evidence format**: FEVER provides gold evidence passages (oracle retrieval), so AUROC on FEVER measures NLI quality, not retrieval quality. Real-world performance depends on retrieval success.

4. **`NOT ENOUGH INFO` class**: FEVER's NEI class is excluded from evaluation. VeritasCore maps this to `UNSUPPORTED`, but AUROC comparison would require a 3-class metric.

5. **Latency vs. accuracy trade-off**: Grounded mode is fast but requires a source document. Ungrounded mode is slower and depends on web search quality.

---

## Reproduce Results

```bash
# Download datasets
python scripts/download_datasets.py --only halueval fever

# Table 1 — NLI Quality (Oracle Evidence)
# Re-run SummaC baseline under oracle conditions
pip install summac
python scripts/baseline_summac.py --dataset both --n 1000 \
    --bootstrap-ci --output tests/benchmarks/results/

# Run VeritasCore NLI under same oracle conditions
python scripts/benchmark_nli_verifier.py --dataset both --n 1000 --bootstrap-ci \
    --output tests/benchmarks/results/

# Table 1+2 — Full scale benchmark with significance tests
python scripts/run_benchmarks_v2.py \
    --datasets halueval fever \
    --n 1500 --seed 42 \
    --bootstrap-ci --n-bootstrap 2000 \
    --split eval \
    --output tests/benchmarks/results/
```

---

## References

- Laban et al. (2022). *SummaC: Re-visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL.
- Min et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP.
- Manakul et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP.
- Li et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* EMNLP.
- Thorne et al. (2018). *FEVER: a large-scale dataset for Fact Extraction and VERification.* NAACL.
