# Literature Review — Hallucination Detection in Large Language Models

> **VeritasCore Phase 0 Deliverable**  
> Status: Complete  
> Scope: Methods, benchmarks, and positioning for post-hoc LLM fact verification

---

## 1. Hallucination Taxonomy

### 1.1 Intrinsic vs. Extrinsic Hallucinations

Ji et al. (2023) — *Survey of Hallucination in Natural Language Generation* — define two primary categories:

**Intrinsic hallucinations** occur when the generated output contradicts information present in the source input. Example: a summarizer claiming "the report was filed on Monday" when the source document says "Tuesday." These are verifiable against a provided context and are directly addressed by VeritasCore's **Grounded Verification** mode (Phase 2).

**Extrinsic hallucinations** are unverifiable against the source — the model introduces information that is neither present nor contradicted. These require external knowledge retrieval, addressed by VeritasCore's **Ungrounded Verification** mode (Phase 3).

### 1.2 Faithfulness vs. Factuality

Maynez et al. (2020) — *On Faithfulness and Factuality in Abstractive Summarization* — distinguish:

- **Faithfulness**: Does the output accurately reflect the provided source document?
- **Factuality**: Is the output accurate with respect to world knowledge?

VeritasCore addresses both: grounded mode handles faithfulness; ungrounded mode handles factuality.

### 1.3 Closed-Domain vs. Open-Domain

- **Closed-domain**: Verification against a bounded knowledge source (a document, database, or set of retrieved passages). Well-suited to NLI-based methods.
- **Open-domain**: Verification against unbounded world knowledge. Requires retrieval augmentation.

VeritasCore supports both via its `VerificationMode` enum: `GROUNDED` for closed-domain, `UNGROUNDED` for open-domain.

---

## 2. Detection Methods

### 2.1 NLI-Based Methods

Natural Language Inference (NLI) models classify the relationship between a premise (evidence) and hypothesis (claim) as *entailment*, *contradiction*, or *neutral*. Applied to hallucination detection:

**Q² (Honovich et al., 2022)** — *Q²: Evaluating Factual Consistency of Knowledge-Grounded Dialogues via Question Generation and Question Answering* — generates questions from the reference, answers them from the response, then checks consistency. Achieves strong performance on dialogue faithfulness.

**SummaC (Laban et al., 2022)** — *SummaC: Re-Visiting NLI-Based Models for Inconsistency Detection in Summarization* — applies NLI at the sentence level between source and summary sentences. Introduces a "NLI aggregation" strategy that significantly outperforms document-level NLI.

**Key insight for VeritasCore**: Sentence-level NLI significantly outperforms document-level NLI. Phase 2 applies NLI at the **atomic claim level** (post-decomposition), which is finer-grained than either Q² or SummaC.

**Chosen NLI model**: `cross-encoder/nli-deberta-v3-base` — achieves state-of-the-art NLI performance at a size feasible for 8GB VRAM (see Phase 2).

### 2.2 Retrieval-Based Methods

**FActScore (Min et al., 2023)** — *FActScore: Fine-Grained Atomic Evaluation of Factual Precision in Long Form Text Generation* — decomposes text into atomic facts, retrieves Wikipedia passages per fact, and applies NLI to check each fact against retrieved evidence. This paper directly motivates VeritasCore's architecture: claim decomposition → retrieval → NLI per claim.

**RARR (Gao et al., 2023)** — *RARR: Researching and Revising What Language Models Say, Using Language Models* — retrieves evidence via web search, then uses an LLM to identify discrepancies. Introduces the idea of automated revision, not just detection. Informs Phase 3's retrieval verification design.

**Key insight for VeritasCore**: FActScore's atomic decomposition strategy is the most direct precursor to Phase 1 (Claim Decomposition). VeritasCore generalizes FActScore by supporting non-Wikipedia retrieval and adding fusion scoring.

### 2.3 Self-Consistency Methods

**SelfCheckGPT (Manakul et al., 2023)** — *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models* — samples multiple generations from the same LLM and checks internal consistency: facts that are consistent across generations are likely true, while inconsistent facts are likely hallucinated. Works without any external knowledge source.

**Key insight for VeritasCore**: Self-consistency is a complementary signal, particularly valuable when no context or search is available. Phase 4 (Semantic Consistency) implements a variant of this approach. The consistency score is one of three signals fused in Phase 5.

**Limitation**: Requires multiple LLM generations, which increases latency and cost. Phase 4 uses embeddings to approximate consistency without re-querying the LLM.

### 2.4 Ensemble / Fusion Methods

**Chen et al. (2023)** — *FAKING IT: Fooling LLM-based Evaluations with Fake Responses* — demonstrates that individual signals are gameable; multi-signal fusion is more robust.

**HaDes (Liu et al., 2022)** — *Token-level Direct Inference for Hallucination Detection* — combines token probability signals with NLI scores. Shows that calibrated fusion of heterogeneous signals outperforms any single signal.

**Key insight for VeritasCore**: Phase 5's fusion model combines three signals (NLI, retrieval, consistency) using logistic regression → XGBoost, directly motivated by these ensemble results.

---

## 3. Claim Decomposition

### 3.1 Atomic Fact Extraction

FActScore (Min et al., 2023) defines an *atomic fact* as a single, independently verifiable claim that cannot be further decomposed without losing meaning. This definition guides Phase 1's decomposer prompt engineering.

**Criteria for atomic claims**:
1. Self-contained (no required antecedents from surrounding text)
2. Binary-verifiable (can be evaluated as true or false)
3. Minimally specific (no unnecessary conjunctions)

### 3.2 Decomposition Approaches

Three strategies exist:

| Approach | Method | Quality | Cost |
|----------|--------|---------|------|
| **LLM-based** | Prompt an LLM to decompose | High | Medium (local) |
| **Rule-based** | Parse with dependency trees + heuristics | Medium | Low |
| **Fine-tuned** | Train a decomposition model | Highest | High |

VeritasCore Phase 1 uses **LLM-based decomposition** (Phi-3-mini locally) as the primary approach, with a rule-based fallback. This follows FActScore's approach but with a local model instead of GPT.

### 3.3 Decomposition Prompt Strategy

Based on (Min et al., 2023) and empirical findings from (Chern et al., 2023 — *FacTool*):

- Provide 3–5 in-context examples of decomposition
- Instruct the model to maintain co-reference ("Einstein" not "he")
- Request one claim per line for easy parsing
- Validate atomicity by checking for conjunctions ("and", "but") in output

---

## 4. Benchmarks

### 4.1 HaluEval

**Paper**: Ji et al. (2023) — *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*

**Description**: Pairs of (document, hallucinated response, non-hallucinated response) for QA, dialogue, and summarization tasks. 10K samples per task.

**Why primary**: Binary labels at the response level, making it directly applicable to VeritasCore's overall verdict output.

**Metric**: Binary classification accuracy, F1.

**Baseline**: GPT-3.5 achieves ~62% accuracy; VeritasCore target: ≥65%.

### 4.2 FEVER

**Paper**: Thorne et al. (2018) — *FEVER: A Large-Scale Dataset for Fact Extraction and VERification*

**Description**: 185K Wikipedia-based claim-evidence pairs labeled as SUPPORTS, REFUTES, or NOT ENOUGH INFO. The three-way label directly maps to VeritasCore's `Verdict` enum.

**Why primary**: Large scale, established baselines, direct mapping to VeritasCore's output schema.

**Metric**: FEVER score (recall + label accuracy), AUROC.

**Baseline**: DeBERTa-based models achieve ~90% on dev set; VeritasCore NLI component target: ≥80% (lower because we use claims as hypotheses, not gold FEVER claims).

### 4.3 TruthfulQA

**Paper**: Lin et al. (2022) — *TruthfulQA: Measuring How Models Tell the Truth*

**Description**: 817 questions across 38 categories where LLMs frequently hallucinate. Evaluates truthfulness of generative outputs.

**Why secondary**: Tests the full pipeline end-to-end, not just individual components.

**Metric**: % truthful, % informative.

### 4.4 Evaluation Metrics

For binary hallucination detection (HaluEval):
- **AUROC**: Primary metric — threshold-independent, standard for detection tasks
- **F1**: Important when classes are imbalanced
- **Precision/Recall**: Separately tracked for false positive analysis

For three-way verification (FEVER-style):
- **Macro-F1**: Equal weight across SUPPORTS/REFUTES/NEI
- **Accuracy**: Overall label accuracy

---

## 5. Gaps & VeritasCore's Position

### Gap 1: Single-Signal Methods

Most existing methods rely on a single signal (NLI only, retrieval only, or consistency only). VeritasCore fuses all three, which ensemble literature shows consistently outperforms individual signals.

### Gap 2: Paid API Dependencies

FActScore requires OpenAI API. RARR requires GPT-3. SelfCheckGPT works with any LLM but most implementations use paid APIs. **VeritasCore is fully local and free**, targeting consumer hardware (8GB VRAM).

### Gap 3: Model Lock-In

Most methods are tied to specific LLMs (GPT-3, GPT-4). VeritasCore is **model-agnostic**: it verifies the *output* of any LLM without requiring access to the LLM itself.

### Gap 4: Span-Level Explainability

FActScore provides per-claim verdicts but no span linking. SummaC provides sentence-level attribution. **VeritasCore links every verdict to its exact source span in the original response**, enabling highlighted UI and developer debugging.

### Gap 5: Domain Profiles

Existing tools treat all domains equally. Medical and legal hallucinations have higher stakes and different evidence standards. **VeritasCore's domain profile system** (Phase 6) allows configurable confidence thresholds and evidence requirements per domain.

---

## References

1. Ji, Z. et al. (2023). Survey of Hallucination in Natural Language Generation. *ACM Computing Surveys*.
2. Maynez, J. et al. (2020). On Faithfulness and Factuality in Abstractive Summarization. *ACL 2020*.
3. Honovich, O. et al. (2022). Q²: Evaluating Factual Consistency of Knowledge-Grounded Dialogues. *EMNLP 2022*.
4. Laban, P. et al. (2022). SummaC: Re-Visiting NLI-Based Models for Inconsistency Detection. *TACL*.
5. Min, S. et al. (2023). FActScore: Fine-Grained Atomic Evaluation of Factual Precision. *EMNLP 2023*.
6. Gao, T. et al. (2023). RARR: Researching and Revising What Language Models Say. *ACL 2023*.
7. Manakul, P. et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection. *EMNLP 2023*.
8. Liu, Y. et al. (2022). Token-level Direct Inference for Hallucination Detection. *EMNLP Findings 2022*.
9. Thorne, J. et al. (2018). FEVER: A Large-Scale Dataset for Fact Extraction and VERification. *NAACL 2018*.
10. Lin, S. et al. (2022). TruthfulQA: Measuring How Models Tell the Truth. *ACL 2022*.
11. Chern, E. et al. (2023). FacTool: Factuality Detection in Generative AI. *arXiv:2307.13528*.
