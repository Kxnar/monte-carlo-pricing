"""Shared evidence records, semantic seeds and reproducibility metadata."""
from datetime import datetime, timezone
from pathlib import Path
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import subprocess
import warnings
import numpy as np


TRIAL_FIELDS = (
    "schema_version", "suite", "case_id", "split", "workload_id", "valuation_index",
    "method", "group", "repeat", "n_requested", "independent_samples", "path_evaluations",
    "payoff_evaluations", "fixing_evaluations", "quantity", "estimate", "standard_error",
    "reference", "reference_standard_error", "reference_kind", "covered_95",
    "combined_uncertainty_covered_95", "seed_key", "tuning_seed_key", "production_seconds",
    "tuning_seconds", "setup_seconds", "decision_seconds", "cumulative_seconds", "peak_memory_bytes",
    "memory_kind", "proposal_id", "retuned", "coefficient_id", "block_size", "status",
)
_COUNT_FIELDS = {"schema_version", "valuation_index", "group", "repeat", "n_requested", "independent_samples",
                 "path_evaluations", "payoff_evaluations", "fixing_evaluations", "peak_memory_bytes", "block_size"}
_NUMERIC_FIELDS = {"estimate", "standard_error", "reference", "reference_standard_error", "production_seconds",
                   "tuning_seconds", "setup_seconds", "decision_seconds", "cumulative_seconds"}
_NONNEGATIVE_FIELDS = _NUMERIC_FIELDS-{"estimate", "reference"}
_BOOL_FIELDS = {"covered_95", "combined_uncertainty_covered_95", "retuned"}


def trial_record(**fields):
    unknown = set(fields)-set(TRIAL_FIELDS)
    if unknown:
        raise ValueError(f"Unexpected trial fields: {sorted(unknown)}")
    record = dict.fromkeys(TRIAL_FIELDS)
    record.update(schema_version=1, status="ok")
    record.update(fields)
    for key, value in record.items():
        if value is None:
            continue
        if key in _COUNT_FIELDS:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer.")
            record[key] = int(value)
        elif key in _NUMERIC_FIELDS:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite numeric data.")
            if key in _NONNEGATIVE_FIELDS and value < 0:
                raise ValueError(f"{key} must be nonnegative.")
            record[key] = float(value)
        elif key in _BOOL_FIELDS:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{key} must be boolean.")
            record[key] = bool(value)
    return record


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON object: {type(value).__name__}")


def stable_seed(root, *semantic_keys):
    """Canonical structured keys, with 128-bit entropy for NumPy generators."""
    payload = json.dumps([root, *semantic_keys], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False, default=_json_default).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def environment_metadata():
    versions = {}
    for name in ("numpy", "scipy", "QuantLib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    config = io.StringIO()
    with contextlib.redirect_stdout(config), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Install `pyyaml`.*", category=UserWarning)
        np.show_config()
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    sources = [p for folder in ("mcpricing", "examples", "tests") for p in (root/folder).rglob("*.py")]
    sources += [p for p in (root/"docs").rglob("*") if p.is_file()]
    sources += [root/name for name in ("pyproject.toml", "requirements.txt", "requirements-reproduce.txt",
                                     "requirements-workbench-reproduce.txt", "README.md", ".github/workflows/tests.yml")
                if (root/name).exists()]
    hashes = {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(sources)}
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "versions": versions, "os": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "blas": config.getvalue(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "source_hashes": hashes,
            "timing_policy": "perf_counter wall time; production excludes tuning, references and traced-memory passes"}


def refuse_nonempty_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=_json_default)+"\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else list(TRIAL_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
