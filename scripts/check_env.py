"""Environment doctor for JISR. Verifies the full local stack so failures show
up here (once) instead of deep inside the pipeline. Run after install:

    python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import pathlib
import shutil
import socket
import sys

# Make the repo root importable so `src.*` resolves when run as a script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _ok(label, val):
    print(f"  [OK]   {label}: {val}")


def _warn(label, val):
    print(f"  [WARN] {label}: {val}")


def check_python_pkgs():
    print("Python packages")
    # ragas needs the langchain-1.x compatibility shim before import.
    try:
        from src.eval import _compat  # noqa: F401
    except Exception:
        pass
    for mod in ["langgraph", "langchain", "FlagEmbedding", "ragas",
                "psycopg", "pgvector", "ollama", "torch", "transformers"]:
        try:
            m = importlib.import_module(mod)
            _ok(mod, getattr(m, "__version__", "imported"))
        except Exception as e:  # noqa: BLE001
            _warn(mod, f"import failed: {e}")


def check_gpu():
    print("\nGPU / inference")
    try:
        import torch
        if torch.cuda.is_available():
            _ok("CUDA", f"{torch.cuda.get_device_name(0)} "
                        f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)")
        else:
            _warn("CUDA", "torch reports no GPU — installed CPU build? "
                          "For RTX 5070 reinstall torch with the CUDA index "
                          "(see scripts/check_env.py notes).")
    except Exception as e:  # noqa: BLE001
        _warn("torch", e)


def check_binaries():
    print("\nExternal binaries (OCR)")
    for exe in ["tesseract", "pdftoppm"]:  # pdftoppm == Poppler
        path = shutil.which(exe)
        (_ok if path else _warn)(exe, path or "not on PATH")


def check_services():
    print("\nServices")
    for label, host, port in [("Postgres", "localhost", 5432),
                              ("Ollama", "localhost", 11434)]:
        s = socket.socket()
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            _ok(label, f"reachable on {host}:{port}")
        except Exception:
            _warn(label, f"not reachable on {host}:{port}")
        finally:
            s.close()


if __name__ == "__main__":
    print("=== JISR environment doctor ===\n")
    check_python_pkgs()
    check_gpu()
    check_binaries()
    check_services()
    print("\nNOTE: CUDA torch for RTX 5070 (Blackwell, sm_120 needs CUDA 12.8):")
    print("  pip uninstall -y torch")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu128")
