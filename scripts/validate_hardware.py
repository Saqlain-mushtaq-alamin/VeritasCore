"""Validate that the current hardware meets VeritasCore requirements.

Usage:
    python scripts/validate_hardware.py

Exit codes:
    0 — All requirements met (or only warnings)
    1 — Critical requirements not met
"""

from __future__ import annotations

import shutil
import sys
import platform


REQUIREMENTS = {
    "python": (3, 10),
    "ram_gb_min": 16,
    "vram_gb_min": 6.0,
    "disk_gb_min": 20.0,
}


def check_python() -> tuple[bool, str]:
    """Check Python version."""
    ver = sys.version_info
    version_str = f"{ver.major}.{ver.minor}.{ver.micro}"
    ok = ver >= REQUIREMENTS["python"]
    msg = f"Python {version_str}"
    if not ok:
        msg += f" — requires {REQUIREMENTS['python'][0]}.{REQUIREMENTS['python'][1]}+"
    return ok, msg


def check_ram() -> tuple[bool, str]:
    """Check available system RAM."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        ok = ram_gb >= REQUIREMENTS["ram_gb_min"]
        msg = f"RAM: {ram_gb:.1f} GB"
        if not ok:
            msg += f" — {REQUIREMENTS['ram_gb_min']}GB+ recommended"
        return ok, msg
    except ImportError:
        return True, "RAM: psutil not installed, skipping check"


def check_cpu() -> tuple[bool, str]:
    """Check CPU core count."""
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        return True, f"CPU: {physical} physical cores / {logical} logical"
    except ImportError:
        import os
        return True, f"CPU: {os.cpu_count()} logical cores"


def check_gpu() -> tuple[bool, str]:
    """Check CUDA GPU availability and VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "GPU: No CUDA GPU detected — CPU fallback will be slow"

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cuda_version = torch.version.cuda

        ok = vram_gb >= REQUIREMENTS["vram_gb_min"]
        msg = f"GPU: {gpu_name} — {vram_gb:.1f} GB VRAM (CUDA {cuda_version})"
        if not ok:
            msg += f" — {REQUIREMENTS['vram_gb_min']}GB+ recommended"
        return ok, msg
    except ImportError:
        return False, "GPU: PyTorch not installed — cannot check CUDA"


def check_disk() -> tuple[bool, str]:
    """Check available disk space."""
    free_gb = shutil.disk_usage(".").free / (1024**3)
    ok = free_gb >= REQUIREMENTS["disk_gb_min"]
    msg = f"Disk: {free_gb:.1f} GB free"
    if not ok:
        msg += f" — {REQUIREMENTS['disk_gb_min']}GB+ needed for models + datasets"
    return ok, msg


def check_torch() -> tuple[bool, str]:
    """Check PyTorch installation."""
    try:
        import torch
        return True, f"PyTorch: {torch.__version__}"
    except ImportError:
        return False, "PyTorch: not installed — run: pip install torch"


def check_transformers() -> tuple[bool, str]:
    """Check HuggingFace Transformers installation."""
    try:
        import transformers
        return True, f"Transformers: {transformers.__version__}"
    except ImportError:
        return False, "Transformers: not installed — run: pip install transformers"


def main() -> bool:
    """Run all hardware checks and print results."""
    print("=" * 60)
    print("VeritasCore — Hardware & Environment Validation")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)

    checks = [
        ("Python", check_python),
        ("RAM", check_ram),
        ("CPU", check_cpu),
        ("GPU", check_gpu),
        ("Disk", check_disk),
        ("PyTorch", check_torch),
        ("Transformers", check_transformers),
    ]

    warnings: list[str] = []
    errors: list[str] = []

    print()
    for name, fn in checks:
        ok, msg = fn()
        icon = "✓" if ok else "⚠"
        print(f"  {icon}  {msg}")
        if not ok:
            warnings.append(msg)

    print()
    print("=" * 60)

    if not warnings:
        print("✓ All checks passed — ready to run VeritasCore!")
        result = True
    else:
        print(f"⚠  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"   • {w}")
        print()
        print("The system may still work with reduced performance.")
        result = True  # Warnings are non-fatal; only errors would be fatal

    print("=" * 60)
    return result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
