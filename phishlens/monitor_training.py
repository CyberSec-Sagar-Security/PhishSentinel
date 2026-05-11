"""
PhishLens Live Training Monitor
================================
Run this in a SEPARATE terminal while training is running:

    python monitor_training.py

Shows:
  - Current training step (parsed from latest log line)
  - GPU utilisation % and memory used (NVIDIA RTX 2000 Ada)
  - CPU utilisation %
  - RAM used
  - Elapsed time
  - Estimated completion

Refreshes every 2 seconds. Press Ctrl+C to stop.
"""

import os
import sys
import time
import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# ── Resolve log file ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"

def _latest_log() -> Path | None:
    if not LOG_DIR.exists():
        return None
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None

# ── GPU stats via nvidia-smi ──────────────────────────────────────────────────
def _gpu_stats() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "util":    int(parts[0]),
            "mem_mb":  int(parts[1]),
            "mem_max": int(parts[2]),
            "temp":    int(parts[3]),
            "power":   float(parts[4]) if parts[4] != "N/A" else 0.0,
        }
    except Exception:
        return {"util": -1, "mem_mb": 0, "mem_max": 8192, "temp": 0, "power": 0.0}

# ── CPU / RAM stats ───────────────────────────────────────────────────────────
def _sys_stats() -> dict:
    try:
        import psutil
        return {
            "cpu": psutil.cpu_percent(interval=None),
            "ram_gb": psutil.virtual_memory().used / 1024**3,
            "ram_total": psutil.virtual_memory().total / 1024**3,
        }
    except ImportError:
        return {"cpu": -1, "ram_gb": 0, "ram_total": 32}

# ── Parse latest step from log file ──────────────────────────────────────────
STEP_PATTERNS = [
    (r"\[1/6\]", "Step 1/6 — Loading datasets"),
    (r"\[2/6\]", "Step 2/6 — Splitting train/test"),
    (r"\[3/6\]", "Step 3/6 — Feature extraction (CPU parallel)"),
    (r"Batch TF-IDF transform", "Step 3/6 — Batch TF-IDF (CPU)"),
    (r"Extracting features.*parallel", "Step 3/6 — Parallel feature extraction (CPU, ~10 min)"),
    (r"Embedding cache HIT",  "Step 3/6 — Embedding cache loaded ✓"),
    (r"Feature matrix shape", "Step 3/6 — Feature matrix assembled ✓"),
    (r"Applying SMOTE",       "SMOTE — Oversampling minority class (CPU, ~5 min)"),
    (r"After SMOTE",          "SMOTE — Complete ✓"),
    (r"\[4/6\]", "Step 4/6 — Training Isolation Forest (CPU)"),
    (r"\[5/6\]", "Step 5/6 — Training ML models (GPU ACTIVE)"),
    (r"Training: LR",       "  Step 5 ↳ Logistic Regression (CPU)"),
    (r"Training: RF",       "  Step 5 ↳ Random Forest (CPU, n_jobs=-1)"),
    (r"Training: XGBOOST",  "  Step 5 ↳ XGBoost  ⚡ GPU spike to 50-80% expected"),
    (r"Training: LIGHTGBM", "  Step 5 ↳ LightGBM ⚡ GPU spike to 30-60% expected"),
    (r"Training: CATBOOST", "  Step 5 ↳ CatBoost ⚡ GPU spike to 40-70% expected"),
    (r"\[6/6\]", "Step 6/6 — Evaluating models (CPU)"),
    (r"Model Comparison",   "COMPLETE — Model metrics ready"),
    (r"Training complete",  "✅ TRAINING FINISHED"),
]

def _parse_step(log_path: Path) -> tuple[str, str]:
    """Return (current_step_label, last_log_line)."""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return "Reading log...", ""

    current = "Starting up..."
    last_line = lines[-1] if lines else ""
    for line in lines:
        for pattern, label in STEP_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                current = label
    return current, last_line[-120:] if last_line else ""

# ── Bar renderer ──────────────────────────────────────────────────────────────
def _bar(value: int, max_val: int = 100, width: int = 30, color: str = "") -> str:
    filled = int(width * value / max_val) if max_val > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{value:3d}%"
    return f"{color}[{bar}] {pct}\033[0m"

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def _gpu_color(util: int) -> str:
    if util >= 50: return GREEN
    if util >= 10: return YELLOW
    return RED

def _clear():
    os.system("cls" if os.name == "nt" else "clear")

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print(f"{BOLD}PhishLens Live Training Monitor{RESET} — press Ctrl+C to exit\n")
    time.sleep(1)

    while True:
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        gpu  = _gpu_stats()
        sys_ = _sys_stats()
        log  = _latest_log()

        if log:
            step, last_line = _parse_step(log)
        else:
            step = "No log file found — is training running?"
            last_line = f"Expected log in: {LOG_DIR}"

        _clear()
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}  PhishLens Training Monitor   {DIM}Elapsed: {elapsed_str}{RESET}")
        print(f"{BOLD}{'='*60}{RESET}\n")

        # Current step
        print(f"  {CYAN}{BOLD}Current:{RESET} {step}")
        print(f"  {DIM}Last log: {last_line}{RESET}\n")

        # GPU
        gc = _gpu_color(gpu["util"])
        mem_pct = int(gpu["mem_mb"] / gpu["mem_max"] * 100) if gpu["mem_max"] else 0
        print(f"  {BOLD}GPU — NVIDIA RTX 2000 Ada{RESET}")
        if gpu["util"] >= 0:
            print(f"    Compute  {_bar(gpu['util'], color=gc)}")
            print(f"    VRAM     {_bar(mem_pct, color=CYAN)}  {gpu['mem_mb']}MB / {gpu['mem_max']}MB")
            print(f"    Temp: {gpu['temp']}°C   Power: {gpu['power']:.0f}W")
            if gpu["util"] < 5:
                print(f"    {YELLOW}⚠  GPU idle — feature extraction is CPU-only (normal){RESET}")
            elif gpu["util"] >= 30:
                print(f"    {GREEN}✓  GPU actively training a model{RESET}")
        else:
            print(f"    {RED}nvidia-smi not found in PATH{RESET}")
        print()

        # CPU / RAM
        print(f"  {BOLD}CPU / RAM{RESET}")
        if sys_["cpu"] >= 0:
            cpu_color = GREEN if sys_["cpu"] > 50 else YELLOW
            print(f"    CPU      {_bar(int(sys_['cpu']), color=cpu_color)}")
            ram_pct = int(sys_["ram_gb"] / sys_["ram_total"] * 100)
            print(f"    RAM      {_bar(ram_pct, color=CYAN)}  {sys_['ram_gb']:.1f}GB / {sys_['ram_total']:.0f}GB")
        else:
            print(f"    {DIM}Install psutil for CPU/RAM stats:  pip install psutil{RESET}")
        print()

        # Stage guide
        print(f"  {DIM}─ What to expect ─────────────────────────────────────────{RESET}")
        print(f"  {DIM}  Step 3: CPU at 80-100%, GPU at 0%   (parallel parsing){RESET}")
        print(f"  {DIM}  Step 5 XGBoost/CatBoost: GPU jumps to 30-80%          {RESET}")
        print(f"  {DIM}  Step 5 LightGBM:         GPU jumps to 20-60%          {RESET}")
        print(f"  {BOLD}{'='*60}{RESET}")

        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
