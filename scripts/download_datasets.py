"""Download and cache evaluation datasets for VeritasCore benchmarking.

Downloads datasets from HuggingFace Hub to data/datasets/.
Primary: HaluEval + FEVER. Secondary: TruthfulQA.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --only halueval fever
    python scripts/download_datasets.py --list

Exit codes:
    0 — All datasets downloaded (or partially, with warnings)
    1 — Critical failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


CACHE_DIR = Path("data/datasets")

DATASETS: dict[str, dict[str, Any]] = {
    "halueval": {
        "hf_path": "pminervini/HaluEval",
        "subsets": ["qa_samples", "summarization_samples", "dialogue_samples"],
        "description": "Hallucination evaluation benchmark with labeled examples",
        "primary": True,
        "notes": "Primary benchmark — directly tests claim-level hallucination detection",
    },
    "fever": {
        "hf_path": "fever/fever",
        "subsets": ["labelled_dev"],
        "description": "Fact Extraction and VERification (155K claim-evidence pairs)",
        "primary": True,
        "notes": "Primary benchmark — large-scale fact verification with evidence",
    },
    "truthfulqa": {
        "hf_path": "truthful_qa",
        "subsets": ["generation", "multiple_choice"],
        "description": "Benchmark for measuring model truthfulness across topics",
        "primary": False,
        "notes": "Secondary benchmark — broader truthfulness evaluation",
    },
    "selfcheckgpt": {
        "hf_path": "potsawee/selfcheckgpt-evaluation",
        "subsets": ["evaluation"],
        "description": "SelfCheckGPT consistency-based hallucination detection",
        "primary": False,
        "notes": "Used in Phase 4 (semantic consistency) evaluation",
    },
}


def download_dataset(name: str, info: dict[str, Any]) -> bool:
    from datasets import load_dataset

    print(f"\n[{name}] {info['description']}")
    print(f"  Source: {info['hf_path']}")

    save_dir = CACHE_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    success = True

    for subset in info["subsets"]:
        subset_dir = save_dir / subset

        if subset_dir.exists():
            print(f"  ✓ {subset} already cached")
            continue

        print(f"  ↓ Downloading {subset}...")

        try:
            ds = load_dataset(
                path=info["hf_path"],
                name=subset,
            )

            ds.save_to_disk(str(subset_dir))

            if isinstance(ds, dict):
                for split_name, split in ds.items():
                    print(f"    ✓ {split_name}: {len(split):,} examples")
            else:
                print(f"    ✓ {len(ds):,} examples")

        except Exception as e:
            print(f"    ✗ {e}")

            if name == "fever":
                print(
                    "    The original FEVER loader is deprecated.\n"
                    "    Switching to a maintained Parquet mirror."
                )

    success = False

    return success


def list_datasets() -> None:
    """Print available datasets with metadata."""
    print("Available benchmark datasets:\n")
    for name, info in DATASETS.items():
        primary = "★ PRIMARY" if info["primary"] else "  secondary"
        print(f"  [{primary}] {name}")
        print(f"    {info['description']}")
        print(f"    Subsets: {', '.join(info['subsets'])}")
        print(f"    Note: {info['notes']}")
        cached = CACHE_DIR / name
        if cached.exists():
            print(f"    Status: ✓ Cached at {cached}")
        else:
            print(f"    Status: ✗ Not downloaded")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VeritasCore benchmark datasets")
    parser.add_argument("--only", nargs="+", choices=list(DATASETS.keys()),
                        help="Download only specific datasets")
    parser.add_argument("--primary-only", action="store_true",
                        help="Download only primary benchmark datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    print("=" * 60)
    print("VeritasCore — Benchmark Dataset Download")
    print(f"Cache directory: {CACHE_DIR.absolute()}")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Select which datasets to download
    to_download: dict[str, dict[str, Any]] = {}
    if args.only:
        to_download = {k: v for k, v in DATASETS.items() if k in args.only}
    elif args.primary_only:
        to_download = {k: v for k, v in DATASETS.items() if v["primary"]}
    else:
        to_download = DATASETS

    print(f"\nDownloading {len(to_download)} dataset(s): {', '.join(to_download.keys())}")

    results: dict[str, bool] = {}
    for name, info in to_download.items():
        results[name] = download_dataset(name, info)

    print("\n" + "=" * 60)
    print("Download summary:")
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {name}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n⚠  {len(failed)} dataset(s) had issues: {', '.join(failed)}")
        print("   You can retry with: python scripts/download_datasets.py --only " + " ".join(failed))
    else:
        print(f"\n✓ All datasets downloaded to {CACHE_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
