# PhishLens — GPU Setup Guide (NVIDIA CUDA)

## The Problem

When you run `pip install torch` or `pip install -r requirements.txt`, pip installs
the **CPU-only build of PyTorch** silently, even on a machine with a powerful GPU.
There is no error or warning — it just runs on CPU indefinitely.

The result: the sentence-transformer embedding stage (all-MiniLM-L6-v2) takes
**4–5 hours on CPU** for ~13,000 emails. The same stage takes **10–15 minutes on GPU**.

## Hardware This Guide Was Written For

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 2000 Ada Generation Laptop GPU |
| VRAM | 8 GB dedicated GDDR6 |
| Architecture | Ada Lovelace (CUDA compute capability 8.9) |
| Driver | 32.0.15.9186 (Jan 2026) — supports CUDA 12.x |

## Step-by-Step: Install CUDA PyTorch on Windows

### Prerequisites

- NVIDIA driver ≥ 527.41 (CUDA 12.x support). Check with: `nvidia-smi`
- Python virtual environment activated (`.venv\Scripts\Activate.ps1`)

### Step 1 — Uninstall the CPU-only build

```powershell
.venv\Scripts\pip.exe uninstall torch torchvision torchaudio -y
```

### Step 2 — Install the CUDA 12.1 build

```powershell
.venv\Scripts\pip.exe install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu126
```

> **Why CUDA 12.6?** PyTorch only added Python 3.13 wheels starting with the
> cu124/cu126 builds. The cu121 index has no Python 3.13 wheels and will fail
> with `No matching distribution found for torch`. Your NVIDIA driver (32.0.15.9186)
> fully supports CUDA 12.6.

### Step 3 — Install the rest of the requirements

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

`torch` is intentionally absent from `requirements.txt` to prevent accidentally
overwriting the CUDA build with a CPU build on subsequent `pip install -r` runs.

## Verify the Installation

Run this one-liner to confirm CUDA is visible:

```powershell
.venv\Scripts\python.exe -c "import torch; cuda=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if cuda else 'N/A'; mem=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1) if cuda else 0; print(f'CUDA={cuda}  GPU={name}  VRAM={mem}GB')"
```

**Expected output on this machine:**

```
CUDA=True  GPU=NVIDIA RTX 2000 Ada Generation Laptop GPU  VRAM=8.0GB
```

If you see `CUDA=False` or `VRAM=0.0GB`, stop and do not run training. Re-check
the uninstall/install steps above. A common cause is that `pip` installed the CPU
build again because `torch` appeared in `requirements.txt` on an earlier `pip install -r`.

## Performance Comparison

| Stage | CPU (i7/i9, 8 cores) | GPU (RTX 2000 Ada, 8GB) |
|-------|----------------------|--------------------------|
| Embedding 13,469 emails | ~4–5 hours | ~10–15 minutes |
| Embedding 95,000 emails (full corpus) | ~30–40 hours | ~60–90 minutes |
| XGBoost training (GPU `hist`) | ~20 min | ~3–5 min |
| LightGBM training (GPU) | ~15 min | ~2–3 min |

After the first training run, embeddings are cached to disk in
`data/processed/embedding_cache/`. Subsequent runs load from cache instantly —
both CPU and GPU benefit equally from this. GPU acceleration matters most when
adding new emails that are not yet cached.

## How the Code Selects the Device

The device selection is fully automatic. In `src/features/text_features.py`:

```python
import torch
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EMBEDDING_MODEL = SentenceTransformer(model_name, device=_DEVICE)
```

In `src/models/trainer.py`, each tree-based model checks at instantiation:

```python
import torch
_cuda = torch.cuda.is_available()

# XGBoost
XGBClassifier(..., **{"tree_method": "hist", "device": "cuda"} if _cuda else {})

# LightGBM
LGBMClassifier(..., device="gpu" if _cuda else "cpu")

# CatBoost
CatBoostClassifier(..., task_type="GPU" if _cuda else "CPU")
```

**No code changes are needed on CPU-only machines.** The fallback to `cpu` is
unconditional — if `torch.cuda.is_available()` returns `False`, all models train
on CPU exactly as before.

## CPU-Only Machines

If you are running PhishLens on a machine without an NVIDIA GPU (e.g., a CI server
or a Mac), install the default CPU build:

```powershell
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

Training will work correctly on CPU. Expect the embedding stage to take several
hours for large datasets. Use the `--skip-casis --skip-enron-kaggle` flags on
`download_datasets.py` to keep the dataset small enough to train overnight on CPU.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `CUDA=False` after install | CPU wheel reinstalled | Run uninstall step again, then install from cu126 index |
| `CUDA out of memory` | Batch size too large | Reduce `batch_size=256` to `128` in `_compute_embedding()` |
| LightGBM GPU error on first run | LightGBM GPU plugin not compiled | Set `device="cpu"` for LightGBM only; XGBoost and CatBoost will still use GPU |
| CatBoost `task_type='GPU'` error | CatBoost GPU build not installed | `pip install catboost` reinstalls the GPU-capable build automatically |
| `nvidia-smi` not found | Driver not installed or PATH not set | Reinstall NVIDIA driver from nvidia.com |
