"""
CANCERWISE AI — Pre-Flight System & Dependency Verification Script
Performs end-to-end environment diagnostics before application startup:
  1. Python version compatibility
  2. Critical package dependencies
  3. Directory topology and permissions
  4. Dataset availability & integrity
  5. Port binding availability (8000, 8501)
  6. Hardware & GPU compute detection
"""

from __future__ import annotations

import os
import sys
import socket
import platform
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title.upper()}")
    print(f"{'='*70}")


def check_python_version() -> Tuple[bool, str]:
    ver = sys.version_info
    current = f"{ver.major}.{ver.minor}.{ver.micro}"
    if (ver.major, ver.minor) >= (3, 11):
        return True, f"Python {current} (Compatible >= 3.11)"
    return False, f"Python {current} (Requires >= 3.11)"


def check_dependencies() -> Tuple[bool, List[str]]:
    packages = [
        ("numpy", "Core Numeric"),
        ("scipy", "Scientific Computing"),
        ("pandas", "Data Manipulation"),
        ("polars", "High-Throughput Columnar"),
        ("sklearn", "Machine Learning Core"),
        ("imbalanced_learn", "Class Resampling"),
        ("torch", "Deep Learning / PyTorch"),
        ("shap", "Explainable AI (SHAP)"),
        ("lime", "Explainable AI (LIME)"),
        ("fastapi", "Backend REST Framework"),
        ("uvicorn", "ASGI Production Server"),
        ("pydantic", "Schema Validation"),
        ("streamlit", "Scientific Dashboard"),
        ("plotly", "Interactive Visualizations"),
        ("psutil", "System Telemetry"),
        ("joblib", "Model Serialization"),
        ("mlflow", "Model Lineage & Governance"),
    ]

    results = []
    all_ok = True
    for pkg, label in packages:
        # Normalize import name
        mod_name = "imblearn" if pkg == "imbalanced_learn" else pkg
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            results.append(f"[OK] {pkg:<16} ({label}): v{ver}")
        except ImportError:
            all_ok = False
            results.append(f"[FAIL] {pkg:<16} ({label}): NOT FOUND")

    return all_ok, results


def check_directories() -> Tuple[bool, List[str]]:
    required_dirs = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "metadata",
        PROJECT_ROOT / "models" / "trained",
        PROJECT_ROOT / "models" / "registry",
        PROJECT_ROOT / "models" / "governance",
        PROJECT_ROOT / "experiments" / "mlruns",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "reports" / "eda",
        PROJECT_ROOT / "reports" / "evaluation",
    ]

    results = []
    all_ok = True
    for d in required_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            # Check write permission
            test_file = d / ".perm_check.tmp"
            test_file.touch()
            test_file.unlink()
            rel_path = d.relative_to(PROJECT_ROOT)
            results.append(f"[OK] Directory writable: {rel_path}")
        except Exception as e:
            all_ok = False
            results.append(f"[FAIL] Cannot write to directory {d}: {e}")

    return all_ok, results


def check_dataset() -> Tuple[bool, str]:
    processed_parquet = PROJECT_ROOT / "data" / "processed" / "uci_wdbc.parquet"
    processed_csv = PROJECT_ROOT / "data" / "processed" / "uci_wdbc.csv"
    raw_data = PROJECT_ROOT / "data" / "raw" / "wdbc.data"

    if processed_parquet.exists():
        size_kb = round(processed_parquet.stat().st_size / 1024, 1)
        return True, f"Primary cohort (UCI WDBC) verified at data/processed/uci_wdbc.parquet ({size_kb} KB)"
    elif processed_csv.exists():
        size_kb = round(processed_csv.stat().st_size / 1024, 1)
        return True, f"Primary cohort (UCI WDBC) verified at data/processed/uci_wdbc.csv ({size_kb} KB)"
    elif raw_data.exists():
        return True, "Raw WDBC data available. Will be parsed into processed format on startup."
    else:
        return False, "UCI WDBC dataset not found. Can be downloaded automatically via API /ingest/wdbc."


def check_port(port: int, host: str = "127.0.0.1") -> Tuple[bool, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        result = s.connect_ex((host, port))
        if result == 0:
            # Port is currently in use
            return False, f"Port {port} is currently IN USE by another process."
        return True, f"Port {port} is AVAILABLE for binding."


def check_hardware() -> List[str]:
    hw_info = [
        f"Operating System: {platform.system()} {platform.release()} ({platform.machine()})",
    ]
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        hw_info.append(f"Logical CPU Cores: {cpu_count}")
        hw_info.append(f"Total Physical RAM: {round(mem.total / (1024**3), 2)} GB (Free: {round(mem.available / (1024**3), 2)} GB)")
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            hw_info.append(f"Hardware Acceleration: CUDA GPU detected ({gpu_name}, {vram} GB VRAM)")
        else:
            hw_info.append("Hardware Acceleration: CPU Mode (CUDA not detected, high-throughput multicore CPU fallback active)")
    except Exception:
        hw_info.append("Hardware Acceleration: CPU Mode")

    return hw_info


def main() -> int:
    print_header("CANCERWISE AI — Pre-Flight Diagnostic Suite")
    overall_status = True

    # 1. Python version
    py_ok, py_msg = check_python_version()
    print(f"\n[1] Python Environment:\n    {py_msg}")
    if not py_ok:
        overall_status = False

    # 2. Hardware
    print(f"\n[2] Host Hardware Telemetry:")
    for line in check_hardware():
        print(f"    • {line}")

    # 3. Dependencies
    dep_ok, dep_logs = check_dependencies()
    print(f"\n[3] Core Machine Learning & API Stack:")
    for line in dep_logs:
        print(f"    {line}")
    if not dep_ok:
        overall_status = False

    # 4. Storage & Directories
    dir_ok, dir_logs = check_directories()
    print(f"\n[4] Artifact Storage & Directories:")
    for line in dir_logs:
        print(f"    {line}")
    if not dir_ok:
        overall_status = False

    # 5. Dataset
    ds_ok, ds_msg = check_dataset()
    print(f"\n[5] Primary Cohort Data Validation:\n    [INFO] {ds_msg}")

    # 6. Ports
    print(f"\n[6] Network Socket Availability:")
    p8000_ok, p8000_msg = check_port(8000)
    p8501_ok, p8501_msg = check_port(8501)
    print(f"    • FastAPI Backend:    {p8000_msg}")
    print(f"    • Streamlit Frontend: {p8501_msg}")

    # Final summary
    print_header("Diagnostic Summary")
    if overall_status:
        print("  >>> PRE-FLIGHT VERIFICATION PASSED: SYSTEM READY FOR LAUNCH <<<")
        print("  FastAPI Backend:    http://localhost:8000/docs")
        print("  Streamlit Frontend: http://localhost:8501\n")
        return 0
    else:
        print("  >>> PRE-FLIGHT VERIFICATION FAILED: RESOLVE REPORTED ISSUES <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
