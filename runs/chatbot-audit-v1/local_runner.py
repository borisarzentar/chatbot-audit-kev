"""Run Kev's local-data workflow on Windows without Unix orchestration imports.

Suite downloading/orchestration locks are deliberately unsupported here.
This launcher is only for existing local JSONL and cached models.
"""
import os
import runpy
import sys
import types
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["KEV_CUDA_GRAPHS"] = "0"
os.environ["TRITON_CACHE_DIR"] = str(Path(__file__).resolve().parent / ".triton-cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if sys.platform == "win32":
    def unsupported_lock(*args, **kwargs):
        raise RuntimeError("Unix orchestration locks are unsupported in this local-data launcher")

    fcntl = types.ModuleType("fcntl")
    fcntl.LOCK_EX, fcntl.LOCK_NB = 2, 4
    fcntl.flock = unsupported_lock
    sys.modules["fcntl"] = fcntl

    import psutil
    resource = types.ModuleType("resource")
    resource.RUSAGE_SELF = 0
    resource.getrusage = lambda who: types.SimpleNamespace(
        ru_maxrss=psutil.Process().memory_info().peak_wset / 1024
    )
    sys.modules["resource"] = resource

module = sys.argv.pop(1)
if module == "calibrate_checkpoint":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/calibrate_checkpoint.py"), run_name="__main__")
    raise SystemExit(0)
if module not in {"kev.train", "kev.benchmark"}:
    raise SystemExit("Only kev.train and kev.benchmark are supported")
if module == "kev.benchmark":
    from kev import suite
    from kev.model import training_context
    suite.CONTEXT = {**training_context(4096), "truncate": False}
runpy.run_module(module, run_name="__main__")
