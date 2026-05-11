#!/usr/bin/env python3
"""
Patch torch/_jit_internal.py for Python 3.13 compatibility.

Root cause: Python 3.13 changed inspect.getsource() behaviour for @overload_method
decorated stubs. The stub source is returned without the function body, causing
ast.parse() to raise IndentationError at torch import time.

Fix: Wrap the _check_overload_body() call inside _overload_method() in a
try-except so the overload registration continues even when source inspection
fails. This is safe because _check_overload_body is a validation-only guard;
it does not affect the runtime behaviour of the overloaded methods.

Usage:
    python patch_torch_py313.py
"""
import importlib
import pathlib
import sys


def find_jit_internal() -> pathlib.Path:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        sys.exit("torch is not installed in this environment — run: "
                 "pip install torch --index-url https://download.pytorch.org/whl/cu126")
    torch_dir = pathlib.Path(spec.origin).parent
    target = torch_dir / "_jit_internal.py"
    if not target.exists():
        sys.exit(f"Cannot find {target}")
    return target


OLD = "        _check_overload_body(func)\n"
NEW = (
    "        try:\n"
    "            _check_overload_body(func)\n"
    "        except (SyntaxError, IndentationError):\n"
    "            pass  # Python 3.13: inspect.getsource returns incomplete stubs\n"
)


def patch(path: pathlib.Path) -> None:
    src = path.read_text(encoding="utf-8")

    if NEW.strip() in src:
        print(f"[patch_torch_py313] Already patched: {path}")
        return

    if OLD not in src:
        # Try to locate the function regardless — might have been refactored
        if "_check_overload_body(func)" not in src:
            print(f"[patch_torch_py313] Target string not found in {path}.")
            print("  The torch version installed may have already fixed this issue, "
                  "or the internal layout changed. No patch applied.")
            return
        # Fallback: broader match
        old_bare = "        _check_overload_body(func)"
        if old_bare in src:
            patched = src.replace(old_bare, NEW.rstrip("\n"), 1)
            path.write_text(patched, encoding="utf-8")
            _invalidate_cache(path)
            print(f"[patch_torch_py313] Patched (bare match): {path}")
            return
        print(f"[patch_torch_py313] Could not locate target string. Manual patch needed.")
        return

    patched = src.replace(OLD, NEW, 1)
    path.write_text(patched, encoding="utf-8")
    _invalidate_cache(path)
    print(f"[patch_torch_py313] Successfully patched: {path}")


def _invalidate_cache(py_file: pathlib.Path) -> None:
    """Remove compiled .pyc files so Python loads the patched source."""
    pycache = py_file.parent / "__pycache__"
    stem = py_file.stem
    if pycache.exists():
        for pyc in pycache.glob(f"{stem}.*.pyc"):
            pyc.unlink(missing_ok=True)
            print(f"[patch_torch_py313] Removed cached bytecode: {pyc.name}")


def verify() -> None:
    print("[patch_torch_py313] Verifying: import torch ...", flush=True)
    try:
        import torch  # noqa: F401
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            mem = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
            print(f"[patch_torch_py313] CUDA={cuda}  GPU={name}  VRAM={mem}GB")
        else:
            print(f"[patch_torch_py313] CUDA={cuda}  (GPU not detected — check driver)")
    except Exception as exc:
        print(f"[patch_torch_py313] import torch FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    path = find_jit_internal()
    print(f"[patch_torch_py313] Target: {path}")
    patch(path)
    verify()
