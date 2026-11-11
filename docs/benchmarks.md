# Benchmarks — VeritasCore

> Full evaluation results, methodology, and comparison with published baselines.
> All numbers generated automatically from saved raw experimental runs (commit `ebcb1ef10b9a`).

---

## Datasets

### HaluEval QA

| Property | Value |
|----------|-------|
| Source | [HaluEval](https://github.com/RUCAIBox/HaluEval) (Li et al., 2023) |
| Task | Question-answering hallucination detection |
| Size | 10,000 QA pairs (balanced: 50% hallucinated) |
| Format | `(knowledge_paragraph, question, answer, hallucination_label)` |
| Eval split | Held-out n=1,500 (grid-search on separate n=500; zero train-test overlap) |
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

We re-run the SummaC baseline (Laban et al., 2022) under identical conditions so the comparison is strictly paired (`scripts/baseline_summac.py`).

### Full-Pipeline & Ablation Protocol (Table 2)
> "In real deployment without pre-provided evidence, how well does each system perform?"

Each system runs its complete pipeline, including live evidence retrieval via web search.

---

## Results

### Table 1: NLI Quality (Oracle Evidence)

> **Protocol:** Each system receives identical gold evidence passages. Measures NLI model quality in isolation (not retrieval quality).
> **Both systems evaluated under strictly paired held-out conditions** on RTX 4060 (seed=42, bootstrapped 95% CI, n_bootstrap=10,000 resamples).
> *Git commit:* `ebcb1ef10b9a`

| Method | Scoring Formula | HaluEval AUROC (95% CI) | FEVER AUROC (95% CI) | Evidence | n (HaluEval / FEVER) | Significance vs SummaC |
|--------|:---------------:|:-----------------------:|:--------------------:|:--------:|:--------------------:|:----------------------:|
| SummaC ZS (chunk + ent-only, tuned†) | $\max_c \text{ent}(s_i, c)$ | 0.7150 (0.6884–0.7411) | 0.9557 (0.9437–0.9668) | Oracle | 1,500 / 1,000 | Baseline |
| **VeritasCore NLI (a) Forward-only (Default)** | $\text{fwd}_c - \text{fwd}_e$ | **0.7098 (0.6831–0.7362)** | **0.9721 (0.9622–0.9811)** | Oracle | 1,500 / 1,000 | DeLong p=0.5310 (HaluEval) / **p=3.93e-06** (FEVER) |
| VeritasCore NLI (b) Forward-only (asymmetric) | $\text{fwd}_c - 0.2\text{fwd}_e$ | 0.6926 (0.6652–0.7193) | 0.9713 (0.9614–0.9803) | Oracle | 1,500 / 1,000 | DeLong p=0.0006 (HaluEval) / p=0.2340 (FEVER) |
| VeritasCore NLI (c) Bidirectional | $\text{fwd}_c - 0.2\text{fwd}_e - 0.6\text{rev}_e$ | 0.6968 (0.6698–0.7233) | 0.9678 (0.9568–0.9777) | Oracle | 1,500 / 1,000 | DeLong p=0.0693 (HaluEval) / **p=0.0343** (FEVER) |

*†SummaC baseline tuning: Evaluated 4 variants on the grid split (`sentence` vs `chunk`, `ent-only` vs `ent-minus-contra`). The `chunk + ent-only` variant won on the grid split (Grid AUROC = 0.7302) and was deployed on the held-out evaluation set.*
*Statistical Significance (Forward-only (a) vs SummaC):*
- **FEVER:** DeLong $z = 4.6151$ ($p = 3.93e-06$), Paired Bootstrap $p < 1e-4$ ($n=10,000$).
- **HaluEval QA:** DeLong $z = -0.6265$ ($p = 0.5310$), Paired Bootstrap $p = 0.7375$ ($n=10,000$).

*Paired DeLong Comparison: Forward-only (a) vs. Bidirectional (c):*
- **FEVER:** $\Delta\text{AUROC} = +0.0043$ (95% CI: +0.0003 to +0.0084), DeLong $z = 2.1160$ ($p = 0.0343$), Paired Bootstrap $p = 0.0032$. The advantage of Forward-only (a) over Bidirectional (c) is statistically significant on FEVER.
- **HaluEval QA:** $\Delta\text{AUROC} = +0.0130$ (95% CI: -0.0010 to +0.0270), DeLong $z = 1.8166$ ($p = 0.0693$), Paired Bootstrap $p = 0.0348$.

*Scoring Formula & Latency Trade-off:* The canonical symmetric difference $\text{fwd}_c - \text{fwd}_e$ achieves the highest AUROC on both HaluEval (**0.7098**) and FEVER (**0.9721**), outperforming asymmetric forward-only (0.6926 / 0.9713) and bidirectional reverse-scoring (0.6968 / 0.9678) while running at measured **2.2x inference speedup** over bidirectional cross-encoder verification (14.2 ms vs 32.0 ms per claim on RTX 4060).


---

### Length-Controlled Grounding Analysis

> **Protocol:** Evaluates whether VeritasCore provides evidence-grounded discrimination beyond claim surface artifacts (n=1,500 held-out HaluEval QA samples, grouped by knowledge paragraph).

#### 1. Length-Matched Strata (Tertiles)
Within length-matched strata, variation in claim verbosity is constrained, testing if NLI discriminates hallucinations when length is held constant:

| Stratum (Word Count) | n | Length Alone AUROC | Surface Model AUROC | VeritasCore NLI AUROC |
|---|:---:|:---:|:---:|:---:|
| Short (≤ 19 words) | 518 | 0.6967 (0.6468–0.7436) | 0.7534 (0.7069–0.7956) | **0.6436 (0.5808–0.7035)** |
| Medium (20-29 words) | 502 | 0.5943 (0.5442–0.6415) | 0.6010 (0.5501–0.6494) | **0.7590 (0.7149–0.8006)** |
| Long (≥ 30 words) | 480 | 0.5096 (0.4507–0.5681) | 0.6248 (0.5714–0.6774) | **0.7614 (0.7147–0.8062)** |

*Finding on Medium & Long Claims:* In medium claims (20–29 words), length alone provides weak discrimination (AUROC = **0.5943**), and in long claims (≥30 words), length-alone AUROC collapses toward chance (**0.5096**), while VeritasCore NLI maintains strong evidence-based discrimination (**0.7590** and **0.7614**).

*Negative Finding on Short Claims:* In the short stratum (≤ 19 words, n=518), VeritasCore NLI achieves AUROC = **0.6436** (0.5808–0.7035), which is **strictly worse than length alone (0.6967)** and surface features alone (0.7534). For highly concise QA answers (e.g., 1–3 words), token overlap is sparse, causing surface verbosity heuristics to outperform cross-encoder entailment.

#### 2. Incremental Value Beyond Surface Artifacts
- **Surface Features Alone (5-fold CV):** AUROC = **0.7370**
- **VeritasCore NLI Alone (Forward-only):** AUROC = **0.7098**
- **NLI + Surface Features Combined:** AUROC = **0.7787**
- **Incremental Significance (DeLong test, Combined vs. Surface alone):**
  $$z = 4.3847, \quad p = 1.1617e-05 \quad (p < 10^{-4})$$

#### 3. Clean Baseline Control (FEVER, n=1,000)
On FEVER (where claims are clean Wikipedia declarative statements free of generation length heuristics):
- **Surface Features Alone:** AUROC = **0.5610** (near-random, confirming no artifact shortcut)
- **VeritasCore NLI (Forward-only):** AUROC = **0.9721** (pure semantic entailment)


---

### Table 2: Component Ablation Study (Phase R5)

> **Protocol:** Evaluates the contribution of each signal combination on held-out evaluation sets (n=500 per dataset, 10,000 bootstrap resamples).
> **Fusion Model Specification:** Logistic Regression (`class_weight='balanced'`, `max_iter=1000`, `random_state=42`) trained exclusively on the disjoint 500-sample grid split (`models/fusion/model.joblib`) with real data only. Consistency features were removed following audit. Evaluated on the held-out evaluation split.
> *Git commit:* `ebcb1ef10b9a`

| Configuration | HaluEval QA AUROC (95% CI) | FEVER Validation AUROC (95% CI) |
|---|:---:|:---:|
| **Full Fusion** | **0.7523 (0.7089–0.7941)** | **0.9068 (0.8769–0.9350)** |
| -NLI | 0.7203 (0.6750–0.7626) | 0.7321 (0.6859–0.7780) |
| -Retrieval | 0.7442 (0.7006–0.7865) | 0.8449 (0.8083–0.8800) |
| Surface-only | 0.7338 (0.6894–0.7759) | 0.5364 (0.4857–0.5859) |
| +Consistency | 0.9382 (0.9173–0.9565) | N/A (no query) |

#### Paired Statistical Significance vs. Full Fusion
| Ablated Variant | HaluEval Delta AUROC (p-value) | FEVER Delta AUROC (p-value) |
|---|:---:|:---:|
| -NLI | +0.0322 (p=0.0167) | +0.1746 (p=< 1e-4) |
| -Retrieval | +0.0080 (p=0.1514) | +0.0620 (p=< 1e-4) |
| Surface-only | N/A | N/A |
| +Consistency | N/A | N/A |

#### Signal Direction & Consistency Signal Audit
To prevent inverted signal contamination, the discriminative direction of each raw signal was audited:
- **HaluEval QA:**
  - *NLI Entailment:* AUROC($s$) = **0.3002**, AUROC($1-s$) = **0.6998** (higher entailment indicates supported truth; $1-s$ indicates hallucination).
  - *Consistency Signal Audit:* Evaluated across four conditions:
    - (a) Query vs. Answer-only similarity: AUROC = **0.9244**
    - (b) Query vs. "Q:..A:.." claim text: AUROC = **0.2880**
    - (c) Paired within-question concordance: Concordance AUROC = **0.9244**
    - (d) Question-shuffled control: AUROC = **0.4746** (collapses to chance)
  - *(Audit Diagnosis)*: In HaluEval QA, ChatGPT-generated hallucinations heavily copy and mirror question vocabulary, whereas HotpotQA human answers are concise entities. Under question-shuffling, consistency AUROC collapses completely to chance (**0.4746**), confirming consistency is an ungrounded conversational artifact rather than factual verification. As required by scientific integrity, **consistency was eliminated from the main production fusion model**.
- **FEVER (Validation):**
  - *NLI Entailment:* AUROC($s$) = **0.0556**, AUROC($1-s$) = **0.9444**.
  - *Question-Answer Consistency:* Treated as **N/A** (FEVER claims are declarative statements without questions; zero artificial queries are injected).

*Empirical Ablation Summary:*
1. **FEVER Grounding:** Full Fusion (NLI + Retrieval) achieves AUROC = **0.9068 (0.8769–0.9350)**, significantly outperforming -Retrieval / NLI-only (**0.8449 (0.8083–0.8800)**, $p < 10^{-4}$) and -NLI / Retrieval-only (**0.7321 (0.6859–0.7780)**, $p < 10^{-4}$).
2. **HaluEval Fusion Performance:** Full Fusion achieves AUROC = **0.7523 (0.7089–0.7941)** without consistency features. Ablating retrieval (`-Retrieval`) yields AUROC = **0.7442 (0.7006–0.7865)**, and ablating NLI (`-NLI`) yields AUROC = **0.7203 (0.6750–0.7626)**. Retaining the consistency artifact as an exploratory baseline (`+Consistency`) produces an artificial AUROC of **0.9382 (0.9173–0.9565)**, which collapses to chance when questions are shuffled.


---

### Table 3: Calibration & Reliability Analysis (Phase R7)

> **Protocol:** Evaluated on the fusion model output across held-out evaluation splits (n=500 per dataset).
> **Label Specification:** Binary classification where positive = 1 = hallucinated and negative = 0 = supported.
> **Split Independence:** Platt scaling parameters were fitted strictly on the HaluEval QA grid split (n=500). FEVER was evaluated zero-shot with the HaluEval-fitted calibrator (zero FEVER samples used for training or calibration), reconciling with Control 7.

| Dataset | Split Setup | AUROC | F1 | Raw Brier | Calibrated Brier | Brier Gain | ECE Improvement |
|---------|-------------|:-----:|:--:|:---------:|:----------------:|:----------:|:---------------:|
| HaluEval QA | Held-out Eval (n=500) | 0.7523 | 0.7115 | 0.2092 | **0.2060** | **+0.0033** | 0.0782 → **0.0683** |
| FEVER (validation) | Held-out Eval (n=500) | 0.9068 | 0.8401 | 0.1674 | **0.1547** | **+0.0127** | 0.2477 → **0.2282** |

*Calibration diagrams:*
- `docs/calibration/calibration_fusion_halueval_qa.png`
- `docs/calibration/calibration_fusion_fever_validation.png`


---

### Hardware Disclosure (Phase R8 — Paper Section 4.4)

- **CPU:** AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD (12 cores, 24 threads)
- **GPU:** NVIDIA GeForce RTX 4060 (8.0 GB VRAM)
- **RAM:** 31.1 GB
- **OS:** Windows 10.0.26200
- **Software:** Python 3.11.9, PyTorch 2.5.1+cu121, CUDA 12.1

### Table 5a: Per-Component Latency
| Component | Device | Precision | Batch | Avg (ms) | P50 (ms) | P95 (ms) | Throughput (claims/s) | GPU Mem (MB) |
|-----------|--------|:---------:|:-----:|:--------:|:--------:|:--------:|:---------------------:|:------------:|
| Decomposer (Rule-based) | cpu | -- | 1 | 0.2 | 0.2 | 0.2 | 6126.0 | 0 |
| Fusion Scorer | cpu | -- | 1 | 0.1 | 0.1 | 0.2 | 9726.9 | 0 |
| Semantic Consistency Checker | cuda | fp16 | 1 | 4.6 | 4.4 | 5.8 | 219.7 | 96 |
| Retrieval Verifier (Live Search - Cold) | network | -- | 1 | 3964.5 | 3351.8 | 5150.7 | 0.3 | 0 |
| Retrieval Verifier (Warm Cache Hit) | network | -- | 1 | 37.1 | 19.4 | 194.6 | 27.0 | 0 |
| NLI Verifier (Forward-only, FP16) | cuda | fp16 | 1 | 16.9 | 12.4 | 25.9 | 59.2 | 371 |
| NLI Verifier (Forward-only, FP32) | cuda | fp32 | 1 | 14.2 | 12.1 | 23.6 | 70.4 | 723 |
| NLI Verifier (Bidirectional, FP16) | cuda | fp16 | 1 | 26.3 | 25.3 | 35.5 | 38.0 | 371 |
| NLI Verifier (Bidirectional, FP32) | cuda | fp32 | 1 | 32.0 | 30.0 | 49.6 | 31.3 | 723 |
| Decomposer (LLM Phi-3-mini) | cuda | fp16 | 1 | 10152.4 | 10538.2 | 10575.4 | 0.1 | 7588 |

*Note on Latency & Hardware:* DeBERTa-v3 cross-encoder is benchmarked in both **FP32** (14.2 ms forward vs 32.0 ms bidirectional) and **FP16** (16.9 ms forward vs 26.3 ms bidirectional). Phi-3-mini executes in **FP16** precision (10152.4 ms average latency, 7588 MB runtime VRAM) with 100% GPU VRAM allocation and zero CPU offload. Retrieval latency separates cold live network queries (3964.5 ms) from warm disk-cache hits (37.1 ms).

### Table 5b: End-to-End Pipeline Latency
| Pipeline Mode | Device | Claims | Total (ms) | Per-Claim (ms) | GPU Mem (MB) |
|---------------|:------:|:------:|:----------:|:--------------:|:------------:|
| Grounded (RuleDecomposer + NLI Bidi) | cuda | 1 | 21.3 | 21.3 | 739 |
| Grounded (RuleDecomposer + NLI Bidi) | cuda | 3 | 53.8 | 17.9 | 740 |
| Grounded (RuleDecomposer + NLI Bidi) | cuda | 5 | 82.7 | 16.5 | 740 |
| Full Pipeline (Rule + Retrieval + NLI) | cuda | 1 | 22.9 | 22.9 | 736 |
| Full Pipeline (Rule + Retrieval + NLI) | cuda | 3 | 50.5 | 16.8 | 736 |
| Full Pipeline (Rule + Retrieval + NLI) | cuda | 5 | 171.0 | 34.2 | 771 |

*Note on Pipeline Latency:* In multi-claim queries (3 claims), full-pipeline per-claim latency (16.8 ms) is faster than grounded mode (17.9 ms). In grounded mode, verification evaluates against full multi-sentence gold paragraphs (~250 words) with cross-encoder passes over long context. In the full ungrounded pipeline, claims are verified against concise top-1 search snippets (~15–20 words). Because Transformer self-attention complexity scales quadratically $O(N^2)$ with token sequence length, evaluating against concise retrieved evidence is significantly faster per cross-encoder pass than evaluating against dense reference documents.

### Table 5c: Model Cold-Start Loading Times
| Model | Load Time (s) | GPU Memory (MB) |
|-------|:-------------:|:---------------:|
| RuleDecomposer (CPU, no model) | 0.001 | 0 |
| FusionScorer (CPU, no model) | 0.001 | 0 |
| DeBERTa-v3-base (NLI Cross-Encoder) | 0.921 | 735 |
| all-MiniLM-L6-v2 (Embedding Model) | 0.199 | 113 |
| Phi-3-mini-4k-instruct (LLM Decomposer) | 2.416 | 6693 |

### Table 5d: NLI Batch-Size Scaling Analysis
| Batch Size | Avg Per-Claim (ms) | Throughput (claims/s) | GPU Memory (MB) |
|:----------:|:------------------:|:---------------------:|:---------------:|
| 1 | 25.8 | 38.78 | 754 |
| 2 | 10.9 | 92.09 | 778 |
| 4 | 8.4 | 119.36 | 823 |
| 8 | 7.9 | 126.09 | 916 |
| 16 | 8.3 | 119.78 | 1099 |

*Note on Batch Scaling:* Table 5d benchmarks batched forward-pass execution in `NLIVerifier.verify_batch()`. Scaling batch size from 1 to 8 achieves a **3.25x throughput speedup** (38.8 to 126.1 claims/s) while cutting average per-claim latency from 25.8 ms to 7.9 ms with GPU VRAM scaling modestly from 754 MB to 916 MB.


---

## Empirical Controls & Sanity Checks

To confirm methodological validity and eliminate experimental artifacts, seven rigorous empirical controls were conducted (detailed full report in [`docs/empirical_controls_report.md`](empirical_controls_report.md)):

1. **Shuffled-Label AUROC (Null Hypothesis Test):**
   - HaluEval QA: **0.5008 ± 0.0141** (vs. original 0.7098)
   - FEVER: **0.4965 ± 0.0156** (vs. original 0.9721)
   - *Confirmation:* Perfectly centered at 0.50, proving scoring correctness and zero label/sorting contamination.

2. **Hypothesis-Only Baseline (Claim Surface Features Only):**
   - Evaluates a Logistic Regression baseline using only claim surface features (`claim_length`, `claim_word_count`, `has_numbers`, `has_proper_nouns`, `num_entities`) without any evidence.
   - **FEVER:** 5-fold CV AUROC = **0.5549** (near-random, confirming FEVER claims do not leak truth value via surface shortcuts).
   - **HaluEval QA:** 5-fold CV AUROC = **0.7309** (reveals a known dataset artifact in HaluEval where ChatGPT-generated hallucinations are systematically more verbose than human ground truth answers; controlled via length tertiles in Length-Controlled Grounding Analysis).

3. **FEVER Gold Context Audit:**
   - 20 randomly sampled FEVER validation pairs were inspected. In the `copenlu/fever_gold_evidence` benchmark download, `triple[2]` contains the **actual gold sentence text** (not just the Wikipedia page title), explaining the high AUROC (0.9721) of DeBERTa-v3 on gold evidence passages.

4. **Determinism:**
   - Independent passes on duplicate evaluation splits yielded `max_abs_diff = 0.00e+00`, confirming exact bitwise reproducibility under fixed seeds.

5. **LLM Decomposer Fallback & Timeout Behavior:**
   - Evaluated thread-based timeout architecture (`daemon=True`, 60s timeout). Across the hardware evaluation on RTX 4060, **0 timeouts occurred** (maximum observed latency was 15.24s against the 60s limit), and fallback was never invoked.

6. **Holm-Bonferroni Correction:**
   - Applied step-down correction across all ablation comparisons against Full Fusion. Primary signal ablations on FEVER (`-NLI` adjusted p = 0.0003; `-Retrieval` adjusted p = 0.0003) remain highly statistically significant at $\alpha = 0.05$.

7. **Out-of-Domain Transfer Protocol:**
   - All models, feature weights, and Platt calibrators were fitted **strictly on the HaluEval QA grid split ($n=500$)**. FEVER ($n=1,000$) served as a true zero-shot out-of-domain transfer benchmark without any tuning on FEVER data.

