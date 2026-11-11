"""
Comprehensive script comparing old backup JSON values vs new generated JSON values.
Reads all values directly from JSON files on disk.
"""
import json
from pathlib import Path

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

old_dir = Path("tests/benchmarks/results/old_backup")
new_table1_path = Path("tests/benchmarks/results/table1_results.json")
new_sig_path = Path("tests/benchmarks/results/significance_results.json")
new_ablation_path = Path("results/ablation/ablation_results.json")
new_calib_path = Path("docs/calibration/calibration_report.json")
new_latency_path = Path("docs/r8_latency_results.json")
new_length_path = Path("docs/length_control_report.json")
new_audit_path = Path("docs/consistency_audit_report.json")
new_formula_comp_path = Path("tests/benchmarks/results/formula_comparison_report.json")

print("=" * 80)
print("1. TABLE 1: NLI Quality & Scoring Formulas")
print("=" * 80)
old_t1 = load_json(old_dir / "old_table1_results.json")
new_t1 = load_json(new_table1_path)
old_sig = load_json(old_dir / "old_significance_results.json")
new_sig = load_json(new_sig_path)

print("--- OLD TABLE 1 ---")
for formula, datasets in old_t1.items():
    h = datasets.get("halueval", {})
    f = datasets.get("fever", {})
    print(f"Formula '{formula}':")
    print(f"  HaluEval: AUROC={h.get('auroc', 0):.4f} ({h.get('ci_auroc_lower', 0):.4f}-{h.get('ci_auroc_upper', 0):.4f}), F1={h.get('f1', 0):.4f}")
    print(f"  FEVER:    AUROC={f.get('auroc', 0):.4f} ({f.get('ci_auroc_lower', 0):.4f}-{f.get('ci_auroc_upper', 0):.4f}), F1={f.get('f1', 0):.4f}")

print("\n--- NEW TABLE 1 ---")
for formula, datasets in new_t1.items():
    h = datasets.get("halueval", {})
    f = datasets.get("fever", {})
    print(f"Formula '{formula}':")
    print(f"  HaluEval: AUROC={h.get('auroc', 0):.4f} ({h.get('ci_auroc_lower', 0):.4f}-{h.get('ci_auroc_upper', 0):.4f}), F1={h.get('f1', 0):.4f}")
    print(f"  FEVER:    AUROC={f.get('auroc', 0):.4f} ({f.get('ci_auroc_lower', 0):.4f}-{f.get('ci_auroc_upper', 0):.4f}), F1={f.get('f1', 0):.4f}")

print("\n--- PAIRED DELONG TEST: Forward-only (a) vs Bidirectional (c) ---")
if new_formula_comp_path.exists():
    fc = load_json(new_formula_comp_path)
    for ds_name in ["halueval", "fever"]:
        comp = fc[ds_name]["comparison_a_vs_c"]
        print(f"{ds_name.upper()}:")
        print(f"  Delta AUROC: {comp['delta_auroc']:.4f} (95% CI: {comp['ci_diff_lower']:.4f} to {comp['ci_diff_upper']:.4f})")
        print(f"  DeLong z: {comp['delong_z']:.4f}, p: {comp['delong_p_value']:.4e}")
        print(f"  Paired Bootstrap p: {comp['paired_bootstrap_p_value']}")

print("\n" + "=" * 80)
print("2. TABLE 2: Ablation Study & Consistency Audit")
print("=" * 80)
old_ab = load_json(old_dir / "old_ablation_results.json")
new_ab = load_json(new_ablation_path)

print("--- OLD TABLE 2 ABLATION ---")
for item in old_ab.get("ablation_results", {}).get("halueval", []):
    cfg = item["config"]
    m = item["metrics"]
    print(f"  HaluEval {cfg}: AUROC={m['auroc']:.4f} ({m['auroc_ci_lower']:.4f}-{m['auroc_ci_upper']:.4f})")
for item in old_ab.get("ablation_results", {}).get("fever", []):
    cfg = item["config"]
    m = item["metrics"]
    print(f"  FEVER    {cfg}: AUROC={m['auroc']:.4f} ({m['auroc_ci_lower']:.4f}-{m['auroc_ci_upper']:.4f})")

print("\n--- NEW TABLE 2 ABLATION ---")
for item in new_ab.get("ablation_results", {}).get("halueval", []):
    cfg = item["config"]
    m = item["metrics"]
    print(f"  HaluEval {cfg}: AUROC={m['auroc']:.4f} ({m['auroc_ci_lower']:.4f}-{m['auroc_ci_upper']:.4f})")
for item in new_ab.get("ablation_results", {}).get("fever", []):
    cfg = item["config"]
    m = item["metrics"]
    print(f"  FEVER    {cfg}: AUROC={m['auroc']:.4f} ({m['auroc_ci_lower']:.4f}-{m['auroc_ci_upper']:.4f})")

if new_audit_path.exists():
    audit_data = load_json(new_audit_path)
    print("\n--- CONSISTENCY SIGNAL AUDIT (HaluEval QA) ---")
    for k, v in audit_data.get("experiments", {}).items():
        print(f"  {k}: AUROC={v.get('auroc', 0):.4f} - {v.get('description', '')}")

print("\n" + "=" * 80)
print("3. TABLE 3: Calibration & Reliability Analysis")
print("=" * 80)
old_cal = load_json(old_dir / "old_calibration_report.json")
new_cal = load_json(new_calib_path)

print("--- OLD CALIBRATION ---")
for ds, v in old_cal.get("datasets", {}).items():
    print(f"  {ds}: AUROC={v.get('auroc', 0):.4f}, F1={v.get('f1', 0):.4f}, Raw Brier={v.get('raw_brier', 0):.4f}, Cal Brier={v.get('calibrated_brier', 0):.4f}, Cal ECE={v.get('calibrated_ece', 0):.4f}")

print("\n--- NEW CALIBRATION ---")
for ds, v in new_cal.get("datasets", {}).items():
    print(f"  {ds}: AUROC={v.get('auroc', 0):.4f}, F1={v.get('f1', 0):.4f}, Raw Brier={v.get('raw_brier', 0):.4f}, Cal Brier={v.get('calibrated_brier', 0):.4f}, Cal ECE={v.get('calibrated_ece', 0):.4f}")

print("\n" + "=" * 80)
print("4. TABLE 5: Latency & Hardware Benchmarks")
print("=" * 80)
old_lat = load_json(old_dir / "old_r8_latency_results.json")
new_lat = load_json(new_latency_path)

print("--- OLD LATENCIES (Table 5a) ---")
for comp, v in old_lat.get("table_5a_components", {}).items():
    print(f"  {comp:40s}: Avg={v.get('latency_ms_avg', 0):.1f} ms, Throughput={v.get('throughput_claims_per_sec', 0):.1f} c/s, Mem={v.get('gpu_memory_peak_mb', 0)} MB")

print("\n--- NEW LATENCIES (Table 5a) ---")
for comp, v in new_lat.get("table_5a_components", {}).items():
    print(f"  {comp:40s}: Avg={v.get('latency_ms_avg', 0):.1f} ms, P50={v.get('latency_ms_p50', 0):.1f} ms, P95={v.get('latency_ms_p95', 0):.1f} ms, Mem={v.get('gpu_memory_peak_mb', 0)} MB")

print("\n--- OLD TABLE 5d (Batch Scaling) ---")
for b, v in old_lat.get("table_5d_batch_scaling", {}).items():
    print(f"  Batch {b}: Per-claim={v.get('latency_ms_per_claim', 0):.1f} ms, Throughput={v.get('throughput_claims_per_sec', 0):.1f} c/s")

print("\n--- NEW TABLE 5d (Batch Scaling) ---")
for b, v in new_lat.get("table_5d_batch_scaling", {}).items():
    print(f"  Batch {b}: Per-claim={v.get('latency_ms_per_claim', 0):.1f} ms, Throughput={v.get('throughput_claims_per_sec', 0):.1f} c/s")

print("\n" + "=" * 80)
print("5. LENGTH CONTROL ANALYSIS")
print("=" * 80)
old_len = load_json(old_dir / "old_length_control_report.json")
new_len = load_json(new_length_path)

print("--- OLD LENGTH STRATA ---")
for s in old_len.get("strata", []):
    print(f"  {s.get('name')} ({s.get('range')} words): n={s.get('n')}, Length AUROC={s.get('length_auroc', 0):.4f}, NLI AUROC={s.get('nli_auroc', 0):.4f}")

print("\n--- NEW LENGTH STRATA ---")
for s in new_len.get("strata", []):
    print(f"  {s.get('name')} ({s.get('range')} words): n={s.get('n')}, Length AUROC={s.get('length_auroc', 0):.4f}, NLI AUROC={s.get('nli_auroc', 0):.4f}")

if "incremental_value" in new_len:
    inc = new_len["incremental_value"]
    print(f"\nIncremental Significance (DeLong Combined vs Surface): z={inc.get('delong_z', 0):.4f}, p={inc.get('delong_p_value', 0):.4e}")

print("\n" + "=" * 80)
print("6. EMPIRICAL CONTROLS")
print("=" * 80)
controls_path = Path("docs/empirical_controls_report.json")
if controls_path.exists():
    ctrl = load_json(controls_path)
    c1 = ctrl.get("controls", {}).get("1_shuffled_labels", {})
    print(f"Shuffled-label HaluEval: {c1.get('halueval_shuffled_auroc', 0):.4f} +/- {c1.get('halueval_shuffled_std', 0):.4f}")
    print(f"Shuffled-label FEVER:    {c1.get('fever_shuffled_auroc', 0):.4f} +/- {c1.get('fever_shuffled_std', 0):.4f}")
    c2 = ctrl.get("controls", {}).get("2_hypothesis_surface_only", {})
    print(f"Surface-only CV HaluEval: {c2.get('halueval_surface_cv_auroc', 0):.4f}")
    print(f"Surface-only CV FEVER:    {c2.get('fever_surface_cv_auroc', 0):.4f}")
    c4 = ctrl.get("controls", {}).get("4_determinism", {})
    print(f"Determinism max_abs_diff: {c4.get('max_abs_diff', 0):.2e}")
