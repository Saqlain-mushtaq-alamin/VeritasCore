"""Comprehensive latency benchmarking with full hardware disclosure.

Phase R8 (Latency & Hardware Analysis) — produces all numbers required for
Table 5a (per-component), Table 5b (E2E pipeline), Table 5c (cold-start) in
the VeritasCore research paper.

What this script measures
--------------------------
* Per-component latency: RuleDecomposer, NLIVerifier, FusionScorer
* NLI batch-size scaling: batch sizes [1, 2, 4, 8] -> throughput, GPU mem
* End-to-end pipeline latency: grounded mode (NLI) with 1 / 3 / 5 claims
* Cold-start model loading times
* Hardware disclosure info

Design decisions
-----------------
* Warm-up runs are EXCLUDED from all per-claim timing (noted where applied).
* Wall-clock time is measured with time.perf_counter() (highest resolution).
* GPU memory is measured with torch.cuda.max_memory_allocated() after warm-up.
* CPU memory is measured with psutil.Process().memory_info().rss.
* CPU-only fallback is fully supported (no CUDA required).

Usage
------
    # Quick run -- smoke-test only (no model loading, n=5 synthetic claims)
    python scripts/benchmark_latency.py --quick

    # Full benchmark (loads NLI model, n=30 claims per configuration)
    python scripts/benchmark_latency.py

    # Save JSON results for paper
    python scripts/benchmark_latency.py --output docs/r8_latency_results.json

    # Batch scaling only
    python scripts/benchmark_latency.py --mode batch-scaling

    # E2E pipeline only
    python scripts/benchmark_latency.py --mode e2e
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

# Force UTF-8 output on Windows so Unicode symbols don't crash CP1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LatencyResult:
    """Per-component or per-configuration latency result."""

    component: str
    device: str
    precision: str
    batch_size: int
    n_samples: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_per_sec: float  # claims/sec
    gpu_memory_mb: float
    cpu_memory_mb: float
    notes: str = ""

    def as_table_row(self) -> str:
        return (
            f"| {self.component:<30} | {self.device:<10} | {self.precision:<8} "
            f"| {self.batch_size:^5} | {self.avg_ms:>8.1f} | {self.p95_ms:>8.1f} "
            f"| {self.throughput_per_sec:>10.2f} | {self.gpu_memory_mb:>8.0f} |"
        )


@dataclass
class ColdStartResult:
    """Model cold-start (loading) timing."""

    model_name: str
    load_time_s: float
    gpu_memory_mb: float
    cpu_memory_mb: float


@dataclass
class E2EResult:
    """End-to-end pipeline result for a given number of claims."""

    mode: str
    device: str
    n_claims: int
    total_ms: float
    per_claim_ms: float
    gpu_memory_mb: float


@dataclass
class BatchScaleResult:
    """NLI throughput at a given batch size."""

    batch_size: int
    throughput_per_sec: float
    avg_ms_per_claim: float
    gpu_memory_mb: float


@dataclass
class BenchmarkReport:
    """Complete R8 benchmark report."""

    hardware: dict[str, Any]
    timestamp: str
    component_results: list[LatencyResult] = field(default_factory=list)
    cold_start_results: list[ColdStartResult] = field(default_factory=list)
    e2e_results: list[E2EResult] = field(default_factory=list)
    batch_scale_results: list[BatchScaleResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------


def get_hardware_info() -> dict[str, Any]:
    """Collect hardware information for the paper's Section 4.4."""
    import psutil
    import torch

    info: dict[str, Any] = {
        "cpu": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_threads_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "os": f"{platform.system()} {platform.version()}",
        "python": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
        info["gpu_sm_count"] = props.multi_processor_count
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["cudnn_version"] = str(torch.backends.cudnn.version())
    else:
        info["gpu"] = "none (CPU fallback)"
        info["gpu_vram_gb"] = 0.0
        info["cuda_version"] = "N/A"

    try:
        import transformers  # type: ignore[import]
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "not installed"

    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(data: list[float], pct: float) -> float:
    """Return the p-th percentile of sorted data."""
    if not data:
        return 0.0
    s = sorted(data)
    idx = max(0, min(int(len(s) * pct / 100), len(s) - 1))
    return s[idx]


def _gpu_mem_mb() -> float:
    """Current max GPU memory allocated in MB (0 if no CUDA)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return 0.0


def _cpu_mem_mb() -> float:
    """Current RSS memory of this process in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except Exception:
        return 0.0


def _reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _detect_precision() -> str:
    """fp16 if CUDA available, else fp32."""
    return "fp16" if _detect_device() == "cuda" else "fp32"


# ---------------------------------------------------------------------------
# Component benchmarks
# ---------------------------------------------------------------------------


def benchmark_rule_decomposer(n_samples: int = 50) -> LatencyResult:
    """Benchmark RuleDecomposer -- CPU-only, no ML model required."""
    print(f"\n  [RuleDecomposer] Benchmarking {n_samples} decompositions...")

    from veritascore.decomposer.rule_decomposer import RuleDecomposer

    decomposer = RuleDecomposer()

    # Warm-up (excluded from timing)
    decomposer.decompose("The sky is blue. Water is H2O.")

    test_text = (
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889 as the centerpiece of the World's Fair. "
        "The tower stands 330 meters tall and was designed by Gustave Eiffel. "
        "It attracts approximately 7 million visitors annually. "
        "The structure was originally intended to be temporary. "
        "It weighs approximately 7,300 tonnes. "
        "The tower has three public floors with restaurants on the first and second."
    )

    latencies: list[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        decomposer.decompose(test_text)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = mean(latencies)
    throughput = 1000.0 / avg if avg > 0 else float("inf")

    result = LatencyResult(
        component="Decomposer (Rule)",
        device="cpu",
        precision="--",
        batch_size=1,
        n_samples=n_samples,
        avg_ms=avg,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        throughput_per_sec=throughput,
        gpu_memory_mb=0.0,
        cpu_memory_mb=_cpu_mem_mb(),
        notes="Warm-up excluded; CPU-only; no model download required",
    )
    print(f"    avg={result.avg_ms:.2f}ms  p95={result.p95_ms:.2f}ms  throughput={result.throughput_per_sec:.0f}/s")
    return result


def benchmark_fusion_scorer(n_samples: int = 100) -> LatencyResult:
    """Benchmark FusionScorer -- pure Python, no ML model."""
    print(f"\n  [FusionScorer] Benchmarking {n_samples} score computations...")

    from veritascore.core.types import Claim, ClaimVerdict, Verdict, VerificationMode
    from veritascore.scorer.fusion import FusionScorer

    scorer = FusionScorer()
    scorer.load()

    claim = Claim(
        id="bench_f",
        text="The Eiffel Tower is 330m tall.",
        source_span=(0, 30),
        source_text="The Eiffel Tower is 330m tall.",
    )
    verdict = ClaimVerdict(
        claim=claim,
        verdict=Verdict.SUPPORTED,
        confidence=0.9,
        nli_score=0.85,
        contradiction_score=0.05,
        retrieval_score=0.80,
        consistency_score=0.75,
        reason="Supported by NLI.",
        verification_mode=VerificationMode.GROUNDED,
    )

    # Warm-up
    scorer.score_claim(verdict)

    latencies: list[float] = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        scorer.score_claim(verdict)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = mean(latencies)
    throughput = 1000.0 / avg if avg > 0 else float("inf")

    result = LatencyResult(
        component="Fusion Scorer",
        device="cpu",
        precision="--",
        batch_size=1,
        n_samples=n_samples,
        avg_ms=avg,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        throughput_per_sec=throughput,
        gpu_memory_mb=0.0,
        cpu_memory_mb=_cpu_mem_mb(),
        notes="Pure Python; no GPU/model; warm-up excluded",
    )
    print(f"    avg={result.avg_ms:.3f}ms  p95={result.p95_ms:.3f}ms  throughput={result.throughput_per_sec:.0f}/s")
    return result


def benchmark_nli_verifier(
    n_samples: int = 30,
    batch_size: int = 1,
    device: str | None = None,
    precision: str | None = None,
) -> LatencyResult:
    """Benchmark NLIVerifier latency (model loaded fresh, warm-up excluded)."""
    device = device or _detect_device()
    precision = precision or _detect_precision()
    print(
        f"\n  [NLIVerifier] Benchmarking {n_samples} claims "
        f"(device={device}, batch_size={batch_size}, precision={precision})..."
    )

    from veritascore.core.types import Claim
    from veritascore.verifier.nli_verifier import NLIVerifier

    _reset_gpu_peak()
    verifier = NLIVerifier()

    # Warm-up (excluded from timing)
    warmup = Claim(
        id="warmup",
        text="The sky appears blue due to Rayleigh scattering.",
        source_span=(0, 49),
        source_text="The sky appears blue due to Rayleigh scattering.",
    )
    warmup_context = "Light from the sun scatters in the atmosphere, making the sky appear blue."
    verifier.verify([warmup], context=warmup_context)

    # Record GPU memory after warm-up (model already resident in VRAM)
    gpu_mem_after_warmup = _gpu_mem_mb()
    _reset_gpu_peak()

    claims = [
        Claim(
            id=f"bench_{i}",
            text=f"The subject number {i} demonstrates a significant correlation.",
            source_span=(0, 60),
            source_text=f"The subject number {i} demonstrates a significant correlation.",
        )
        for i in range(n_samples)
    ]
    context = (
        "A study of 500 subjects demonstrated significant correlations across all groups. " * 10
    )

    latencies: list[float] = []
    batch_buf: list[Claim] = []
    for idx, claim in enumerate(claims):
        batch_buf.append(claim)
        if len(batch_buf) == batch_size or idx == len(claims) - 1:
            t0 = time.perf_counter()
            verifier.verify(batch_buf, context=context)
            elapsed = (time.perf_counter() - t0) * 1000
            per_claim_ms = elapsed / len(batch_buf)
            latencies.extend([per_claim_ms] * len(batch_buf))
            batch_buf = []

    verifier.unload()

    avg = mean(latencies) if latencies else 0.0
    throughput = 1000.0 / avg if avg > 0 else 0.0

    result = LatencyResult(
        component="NLI Verifier",
        device=device,
        precision=precision,
        batch_size=batch_size,
        n_samples=n_samples,
        avg_ms=avg,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        throughput_per_sec=throughput,
        gpu_memory_mb=gpu_mem_after_warmup,
        cpu_memory_mb=_cpu_mem_mb(),
        notes="Warm-up run excluded; model loaded fresh; GPU mem measured post-warmup",
    )
    print(
        f"    avg={result.avg_ms:.1f}ms  p95={result.p95_ms:.1f}ms  "
        f"throughput={result.throughput_per_sec:.2f}/s  gpu_mem={result.gpu_memory_mb:.0f}MB"
    )
    return result


# ---------------------------------------------------------------------------
# Cold-start benchmarks
# ---------------------------------------------------------------------------


def benchmark_cold_start_nli() -> ColdStartResult:
    """Measure NLI model cold-start loading time (includes first inference)."""
    print("\n  [ColdStart] NLIVerifier model loading...")
    _reset_gpu_peak()
    cpu_before = _cpu_mem_mb()

    t0 = time.perf_counter()
    from veritascore.verifier.nli_verifier import NLIVerifier
    from veritascore.core.types import Claim
    verifier = NLIVerifier()
    warmup = Claim(
        id="cs_warmup",
        text="Paris is the capital of France.",
        source_span=(0, 29),
        source_text="Paris is the capital of France.",
    )
    verifier.verify([warmup], context="Paris is in France.")
    load_time = time.perf_counter() - t0

    gpu_mem = _gpu_mem_mb()
    cpu_mem = _cpu_mem_mb() - cpu_before
    verifier.unload()

    result = ColdStartResult(
        model_name="DeBERTa-v3-base (NLI)",
        load_time_s=load_time,
        gpu_memory_mb=gpu_mem,
        cpu_memory_mb=max(0.0, cpu_mem),
    )
    print(f"    load_time={result.load_time_s:.2f}s  gpu_mem={result.gpu_memory_mb:.0f}MB")
    return result


def benchmark_cold_start_rule_decomposer() -> ColdStartResult:
    """Measure RuleDecomposer cold-start (import + first decompose)."""
    print("\n  [ColdStart] RuleDecomposer loading...")
    cpu_before = _cpu_mem_mb()

    t0 = time.perf_counter()
    from veritascore.decomposer.rule_decomposer import RuleDecomposer
    decomposer = RuleDecomposer()
    decomposer.decompose("Paris is the capital of France.")
    load_time = time.perf_counter() - t0

    cpu_mem = _cpu_mem_mb() - cpu_before
    result = ColdStartResult(
        model_name="RuleDecomposer (CPU, no model)",
        load_time_s=load_time,
        gpu_memory_mb=0.0,
        cpu_memory_mb=max(0.0, cpu_mem),
    )
    print(f"    load_time={result.load_time_s:.4f}s")
    return result


def benchmark_cold_start_fusion() -> ColdStartResult:
    """Measure FusionScorer cold-start (import + load)."""
    print("\n  [ColdStart] FusionScorer loading...")
    cpu_before = _cpu_mem_mb()

    t0 = time.perf_counter()
    from veritascore.scorer.fusion import FusionScorer
    scorer = FusionScorer()
    scorer.load()
    load_time = time.perf_counter() - t0

    cpu_mem = _cpu_mem_mb() - cpu_before
    result = ColdStartResult(
        model_name="FusionScorer (CPU, no model)",
        load_time_s=load_time,
        gpu_memory_mb=0.0,
        cpu_memory_mb=max(0.0, cpu_mem),
    )
    print(f"    load_time={result.load_time_s:.4f}s")
    return result


# ---------------------------------------------------------------------------
# Batch scaling analysis
# ---------------------------------------------------------------------------


def benchmark_batch_scaling(
    batch_sizes: list[int] | None = None,
    n_per_batch: int = 3,
) -> list[BatchScaleResult]:
    """Benchmark NLI throughput across multiple batch sizes.

    Warm-up (1 batch at each size) is excluded from timing.
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]

    print(f"\n  [BatchScaling] NLIVerifier throughput for batch_sizes={batch_sizes}...")

    from veritascore.core.types import Claim
    from veritascore.verifier.nli_verifier import NLIVerifier

    verifier = NLIVerifier()
    context = "Scientific research has demonstrated correlation between numerous variables. " * 20

    results: list[BatchScaleResult] = []

    for bs in batch_sizes:
        _reset_gpu_peak()

        # Warm-up one batch at this size
        wu = [
            Claim(
                id=f"wu_{bs}_{i}",
                text=f"Variable {i} shows a significant effect at batch size {bs}.",
                source_span=(0, 55),
                source_text=f"Variable {i} shows a significant effect.",
            )
            for i in range(bs)
        ]
        verifier.verify(wu, context=context)
        _reset_gpu_peak()

        batch_times: list[float] = []
        for b in range(n_per_batch):
            batch = [
                Claim(
                    id=f"scale_{bs}_{b}_{i}",
                    text=f"Subject {b * bs + i} demonstrates a correlation with factor {i}.",
                    source_span=(0, 60),
                    source_text=f"Subject {b * bs + i} demonstrates a correlation.",
                )
                for i in range(bs)
            ]
            t0 = time.perf_counter()
            verifier.verify(batch, context=context)
            batch_times.append((time.perf_counter() - t0) * 1000)

        avg_batch_ms = mean(batch_times)
        avg_per_claim_ms = avg_batch_ms / bs
        throughput = 1000.0 / avg_per_claim_ms if avg_per_claim_ms > 0 else 0.0
        gpu_mem = _gpu_mem_mb()

        r = BatchScaleResult(
            batch_size=bs,
            throughput_per_sec=throughput,
            avg_ms_per_claim=avg_per_claim_ms,
            gpu_memory_mb=gpu_mem,
        )
        results.append(r)
        print(
            f"    batch_size={bs:>2}  avg_per_claim={avg_per_claim_ms:.1f}ms  "
            f"throughput={throughput:.2f}/s  gpu_mem={gpu_mem:.0f}MB"
        )

    verifier.unload()
    return results


# ---------------------------------------------------------------------------
# End-to-end pipeline benchmarks
# ---------------------------------------------------------------------------


def benchmark_e2e_grounded(
    n_claims_list: list[int] | None = None,
    n_repeats: int = 3,
) -> list[E2EResult]:
    """Measure full grounded pipeline (RuleDecomposer + NLIVerifier + FusionScorer).

    Uses RuleDecomposer (not LLM) so results are reproducible without GPU for
    decomposition. We pre-load RuleDecomposer and inject it into the engine so
    LLMDecomposer is never instantiated. Warm-up (1 run) is excluded from timing.
    """
    if n_claims_list is None:
        n_claims_list = [1, 3, 5]

    device = _detect_device()
    print(f"\n  [E2E-Grounded] Pipeline timing (device={device}, repeats={n_repeats})...")

    from veritascore.core.engine import VeritasCoreEngine
    from veritascore.core.config import EngineConfig, ModelConfig, SearchConfig
    from veritascore.decomposer.rule_decomposer import RuleDecomposer

    config = EngineConfig(
        models=ModelConfig(device=device),
        search=SearchConfig(provider="none"),
        verification_mode="grounded",
    )

    responses_by_count: dict[int, str] = {
        1: "The Eiffel Tower is 330 metres tall.",
        3: (
            "The Eiffel Tower is 330 metres tall. "
            "It is located in Paris, France. "
            "It was built for the 1889 World's Fair."
        ),
        5: (
            "The Eiffel Tower is 330 metres tall. "
            "It is located in Paris, France. "
            "It was built for the 1889 World's Fair. "
            "The tower was designed by Gustave Eiffel. "
            "It weighs approximately 7,300 tonnes."
        ),
    }
    context = (
        "The Eiffel Tower is a wrought-iron lattice tower standing 330 metres tall "
        "on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 "
        "as the entrance arch for the 1889 World's Fair. Gustave Eiffel's engineering "
        "company built the tower. It weighs approximately 7,300 tonnes."
    )

    e2e_results: list[E2EResult] = []

    for n_claims in n_claims_list:
        response = responses_by_count.get(n_claims, responses_by_count[3])
        times_ms: list[float] = []

        engine = VeritasCoreEngine(config=config)
        # Inject RuleDecomposer so LLMDecomposer is never loaded (no GPU for decomp)
        engine._decomposer = RuleDecomposer()

        # Warm-up (excluded)
        _reset_gpu_peak()
        engine.verify(response=response, context=context, mode="grounded")
        _reset_gpu_peak()

        for _ in range(n_repeats):
            t0 = time.perf_counter()
            engine.verify(response=response, context=context, mode="grounded")
            times_ms.append((time.perf_counter() - t0) * 1000)

        gpu_mem = _gpu_mem_mb()
        engine.unload()

        avg_total = mean(times_ms)
        r = E2EResult(
            mode="Grounded (RuleDecomposer + NLI)",
            device=device,
            n_claims=n_claims,
            total_ms=avg_total,
            per_claim_ms=avg_total / max(n_claims, 1),
            gpu_memory_mb=gpu_mem,
        )
        e2e_results.append(r)
        print(
            f"    n_claims={n_claims}  total={avg_total:.0f}ms  "
            f"per_claim={r.per_claim_ms:.0f}ms  gpu_mem={gpu_mem:.0f}MB"
        )

    return e2e_results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def print_hardware_disclosure(hw: dict[str, Any]) -> None:
    """Print the hardware disclosure section for inclusion in the paper."""
    print("\n" + "=" * 70)
    print("HARDWARE DISCLOSURE (Paper Section 4.4 -- Implementation Details)")
    print("=" * 70)
    print()
    print("**Hardware:** All experiments were conducted on a single machine with:")
    print(
        f"  - CPU: {hw.get('cpu', 'unknown')} "
        f"({hw.get('cpu_cores_physical', '?')} cores, {hw.get('cpu_threads_logical', '?')} threads)"
    )
    if hw.get("cuda_available"):
        print(f"  - GPU: {hw.get('gpu', 'unknown')} ({hw.get('gpu_vram_gb', '?'):.1f}GB VRAM)")
    else:
        print("  - GPU: None (CPU-only run)")
    print(f"  - RAM: {hw.get('ram_gb', '?'):.1f} GB")
    print(f"  - OS: {hw.get('os', 'unknown')}")
    print()
    print("**Software:**")
    print(f"  - Python {hw.get('python', '?')}")
    print(f"  - PyTorch {hw.get('pytorch_version', '?')}")
    print(f"  - CUDA {hw.get('cuda_version', 'N/A')}")
    print(f"  - Transformers {hw.get('transformers_version', '?')}")
    print()
    print("**Inference:** NLI and embedding models run in fp32 by default.")
    print("  Decomposer (Phi-3-mini, if used) runs in fp16 to fit within 8GB VRAM.")
    print("  No quantization is applied by default.")
    print()


def print_component_table(results: list[LatencyResult]) -> None:
    """Print Table 5a: Per-Component Latency."""
    print("\n" + "=" * 70)
    print("TABLE 5a: Per-Component Latency")
    print("=" * 70)
    header = (
        "| Component                      | Device     | Precision "
        "| Batch | Avg (ms) | P95 (ms) | Throughput | GPU Mem |"
    )
    sep = (
        "|--------------------------------|------------|----------"
        "|:-----:|:--------:|:--------:|:----------:|:-------:|"
    )
    print(header)
    print(sep)
    for r in results:
        print(r.as_table_row())
    print()


def print_cold_start_table(results: list[ColdStartResult]) -> None:
    """Print Table 5c: Cold-Start Model Loading Times."""
    print("\n" + "=" * 70)
    print("TABLE 5c: Model Cold-Start Loading Times")
    print("=" * 70)
    header = "| Model                              | Load Time (s) | GPU Memory (MB) |"
    sep    = "|------------------------------------|:-------------:|:---------------:|"
    print(header)
    print(sep)
    for r in results:
        print(f"| {r.model_name:<34} | {r.load_time_s:>13.3f} | {r.gpu_memory_mb:>15.0f} |")
    total_load = sum(r.load_time_s for r in results)
    total_gpu = sum(r.gpu_memory_mb for r in results)
    print(f"| {'Total (all)':34} | {total_load:>13.3f} | {total_gpu:>15.0f} |")
    print()


def print_e2e_table(results: list[E2EResult]) -> None:
    """Print Table 5b: End-to-End Pipeline Latency."""
    print("\n" + "=" * 70)
    print("TABLE 5b: End-to-End Pipeline Latency")
    print("=" * 70)
    header = "| Pipeline Mode                     | Device | Claims | Total (ms) | Per-Claim (ms) |"
    sep    = "|-----------------------------------|--------|:------:|:----------:|:--------------:|"
    print(header)
    print(sep)
    for r in results:
        print(
            f"| {r.mode:<33} | {r.device:<6} | {r.n_claims:^6} "
            f"| {r.total_ms:>10.0f} | {r.per_claim_ms:>14.0f} |"
        )
    print()


def print_batch_scaling_table(results: list[BatchScaleResult]) -> None:
    """Print Table 5d: NLI Batch-Size Scaling."""
    print("\n" + "=" * 70)
    print("TABLE 5d: NLI Batch-Size Scaling Analysis")
    print("=" * 70)
    header = "| Batch Size | Avg Per-Claim (ms) | Throughput (claims/s) | GPU Memory (MB) |"
    sep    = "|:----------:|:------------------:|:---------------------:|:---------------:|"
    print(header)
    print(sep)
    for r in results:
        print(
            f"| {r.batch_size:^10} | {r.avg_ms_per_claim:^18.1f} "
            f"| {r.throughput_per_sec:^21.2f} | {r.gpu_memory_mb:^15.0f} |"
        )
    print()


def print_optimization_table() -> None:
    """Print Table 5e: Latency Optimization Opportunities (static reference)."""
    print("\n" + "=" * 70)
    print("TABLE 5e: Latency Optimization Opportunities (not applied by default)")
    print("=" * 70)
    rows = [
        ("fp16 NLI inference",               "~1.5-2x",  "Minimal accuracy loss"),
        ("4-bit GPTQ decomposer",             "~2x",      "May reduce decomposition quality"),
        ("ONNX Runtime for NLI",              "~1.3x",    "One-time conversion effort"),
        ("Batched NLI inference (bs=8)",      "~2-4x",    "Requires more GPU memory"),
        ("Distilled NLI (DeBERTa-small)",     "~3x",      "Accuracy degradation expected"),
        ("Async retrieval (already built)",   "~3x",      "Multi-claim pipeline only"),
    ]
    header = "| Optimization                     | Expected Speedup | Trade-off                         |"
    sep    = "|----------------------------------|:----------------:|-----------------------------------|"
    print(header)
    print(sep)
    for name, speedup, tradeoff in rows:
        print(f"| {name:<32} | {speedup:^16} | {tradeoff:<33} |")
    print()


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase R8: Latency & Hardware Analysis benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["all", "components", "cold-start", "e2e", "batch-scaling", "hardware-only"],
        default="all",
        help="Which benchmarks to run",
    )
    p.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of samples per component benchmark",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON results (e.g. docs/r8_latency_results.json)",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode: n=5, skips model loading for fast CI validation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    quick = args.quick
    n = 5 if quick else args.n

    print("=" * 70)
    print("VeritasCore -- Phase R8: Latency & Hardware Analysis")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode: {args.mode}  |  n={n}  |  quick={quick}")
    print("=" * 70)

    hw = get_hardware_info()
    print_hardware_disclosure(hw)

    report = BenchmarkReport(
        hardware=hw,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    run_all = args.mode == "all"

    if args.mode == "hardware-only":
        print("Hardware disclosure complete. Use --mode all to run benchmarks.")

    # Component benchmarks
    if run_all or args.mode == "components":
        print("\n>>> STEP 8.1: Per-Component Latency Benchmarks")
        rule_result = benchmark_rule_decomposer(n_samples=n)
        report.component_results.append(rule_result)

        fusion_result = benchmark_fusion_scorer(n_samples=n * 3)
        report.component_results.append(fusion_result)

        if not quick:
            nli_result = benchmark_nli_verifier(n_samples=n, batch_size=1)
            report.component_results.append(nli_result)

        print_component_table(report.component_results)

    # Cold-start benchmarks
    if run_all or args.mode == "cold-start":
        print("\n>>> STEP 8.3: Cold-Start Model Loading Times")
        cs_rule = benchmark_cold_start_rule_decomposer()
        report.cold_start_results.append(cs_rule)

        cs_fusion = benchmark_cold_start_fusion()
        report.cold_start_results.append(cs_fusion)

        if not quick:
            cs_nli = benchmark_cold_start_nli()
            report.cold_start_results.append(cs_nli)

        print_cold_start_table(report.cold_start_results)

    # End-to-end
    if run_all or args.mode == "e2e":
        if not quick:
            print("\n>>> STEP 8.2: End-to-End Pipeline Latency")
            e2e = benchmark_e2e_grounded(n_claims_list=[1, 3, 5], n_repeats=3)
            report.e2e_results.extend(e2e)
            print_e2e_table(report.e2e_results)
        else:
            print("\n  [E2E] Skipped in --quick mode (requires NLI model load)")

    # Batch scaling
    if run_all or args.mode == "batch-scaling":
        if not quick:
            print("\n>>> STEP 8.4: Batch-Size Scaling Analysis")
            batch_results = benchmark_batch_scaling(batch_sizes=[1, 2, 4, 8], n_per_batch=3)
            report.batch_scale_results.extend(batch_results)
            print_batch_scaling_table(report.batch_scale_results)
        else:
            print("\n  [BatchScaling] Skipped in --quick mode (requires NLI model load)")

    # Optimization table (static reference)
    if run_all or args.mode in ("components", "hardware-only"):
        print_optimization_table()

    # Save JSON output
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        def _to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in asdict(obj).items()}
            if isinstance(obj, list):
                return [_to_dict(x) for x in obj]
            return obj

        out = {
            "timestamp": report.timestamp,
            "hardware": report.hardware,
            "component_results": _to_dict(report.component_results),
            "cold_start_results": _to_dict(report.cold_start_results),
            "e2e_results": _to_dict(report.e2e_results),
            "batch_scale_results": _to_dict(report.batch_scale_results),
            "notes": report.notes,
        }
        args.output.write_text(json.dumps(out, indent=2, default=str))
        print(f"\n>>> Results saved -> {args.output}")

    print("\n" + "=" * 70)
    print("Phase R8 benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
