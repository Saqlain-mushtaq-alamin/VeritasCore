"""
Standalone script to read and compare old vs new benchmark numbers directly from JSON files.
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

print("=" * 80)
print("1. TABLE 1: NLI Quality & Scoring Formulas")
print("=" * 80)
old_t1 = load_json(old_dir / "old_table1_results.json")
new_t1 = load_json(new_table1_path)

print(f"Old Table 1 keys: {list(old_t1.keys())}")
print(f"New Table 1 keys: {list(new_t1.keys())}")

for k in ["summac_zs", "veritas_nli_default", "veritas_nli_bidi", "veritas_nli_formula_b"]:
    if k in old_t1:
        o = old_t1[k]
        print(f"OLD {k}: HaluEval={o.get('halueval_auroc')} ({o.get('halueval_ci')}), FEVER={o.get('fever_auroc')} ({o.get('fever_ci')})")
    if k in new_t1:
        n = new_t1[k]
        print(f"NEW {k}: HaluEval={n.get('halueval_auroc')} ({n.get('halueval_ci')}), FEVER={n.get('fever_auroc')} ({n.get('fever_ci')})")

print("\nDeLong Comparison Forward-only (a) vs Bidirectional (c):")
new_formula_comp = load_json(Path("tests/benchmarks/results/formula_comparison_report.json"))
print(json.dumps(new_formula_comp, indent=2))

print("=" * 80)
print("2. TABLE 2: Ablation Study & Consistency Audit")
print("=" * 80)
old_ab = load_json(old_dir / "old_ablation_results.json")
new_ab = load_json(new_ablation_path)

print(f"Old Ablation runs count: {len(old_ab)}")
print(f"New Ablation runs count: {len(new_ab)}")

print("\nOld Ablation Summary:")
for r in old_ab:
    cfg = r.get("config", {})
    short = cfg.get("short", cfg.get("name"))
    print(f"  {short}: HaluEval={r.get('halueval_auroc')}, FEVER={r.get('fever_auroc')}")

print("\nNew Ablation Summary:")
for r in new_ab:
    cfg = r.get("config", {})
    short = cfg.get("short", cfg.get("name"))
    print(f"  {short}: HaluEval={r.get('halueval_auroc')} ({r.get('halueval_ci')}), FEVER={r.get('fever_auroc')} ({r.get('fever_ci')})")

if new_audit_path.exists():
    audit_data = load_json(new_audit_path)
    print("\nConsistency Signal Audit (HaluEval QA):")
    for k, v in audit_data.get("experiments", {}).items():
        print(f"  {k}: AUROC={v.get('auroc')}, Note={v.get('description', '')}")

print("=" * 80)
print("3. TABLE 3: Calibration & Reliability Analysis")
print("=" * 80)
old_cal = load_json(old_dir / "old_calibration_report.json")
new_cal = load_json(new_calib_path)

print("Old Calibration:")
for ds, v in old_cal.get("datasets", {}).items():
    print(f"  {ds}: AUROC={v.get('auroc')}, F1={v.get('f1')}, Raw Brier={v.get('raw_brier')}, Cal Brier={v.get('calibrated_brier')}, Raw ECE={v.get('raw_ece')}, Cal ECE={v.get('calibrated_ece')}")

print("\nNew Calibration:")
for ds, v in new_cal.get("datasets", {}).items():
    print(f"  {ds}: AUROC={v.get('auroc')}, F1={v.get('f1')}, Raw Brier={v.get('raw_brier')}, Cal Brier={v.get('calibrated_brier')}, Raw ECE={v.get('raw_ece')}, Cal ECE={v.get('calibrated_ece')}")

print("=" * 80)
print("4. TABLE 5: Latency & Hardware Benchmarks")
print("=" * 80)
old_lat = load_json(old_dir / "old_r8_latency_results.json")
new_lat = load_json(new_latency_path)

print("Old Latencies (Table 5a):")
for comp, v in old_lat.get("table_5a_components", {}).items():
    print(f"  {comp}: Avg={v.get('latency_ms_avg')}, Throughput={v.get('throughput_claims_per_sec')}, Mem={v.get('gpu_memory_peak_mb')}")

print("\nNew Latencies (Table 5a):")
for comp, v in new_lat.get("table_5a_components", {}).items():
    print(f"  {comp}: Avg={v.get('latency_ms_avg')}, P50={v.get('latency_ms_p50')}, P95={v.get('latency_ms_p95')}, Mem={v.get('gpu_memory_peak_mb')}")

print("\nOld Table 5d (Batching):")
for b, v in old_lat.get("table_5d_batch_scaling", {}).items():
    print(f"  Batch {b}: Per-claim={v.get('latency_ms_per_claim')}, Throughput={v.get('throughput_claims_per_sec')}")

print("\nNew Table 5d (Batching):")
for b, v in new_lat.get("table_5d_batch_scaling", {}).items():
    print(f"  Batch {b}: Per-claim={v.get('latency_ms_per_claim')}, Throughput={v.get('throughput_claims_per_sec')}")

print("=" * 80)
print("5. LENGTH CONTROL REPORT")
print("=" * 80)
old_len = load_json(old_dir / "old_length_control_report.json")
new_len = load_json(new_length_path)

print("Old Strata:")
for s in old_len.get("strata", []):
    print(f"  {s.get('name')} ({s.get('range')}): n={s.get('n')}, Length AUROC={s.get('length_auroc')}, NLI AUROC={s.get('nli_auroc')}")

print("\nNew Strata:")
for s in new_len.get("strata", []):
    print(f"  {s.get('name')} ({s.get('range')}): n={s.get('n')}, Length AUROC={s.get('length_auroc')}, NLI AUROC={s.get('nli_auroc')}")
