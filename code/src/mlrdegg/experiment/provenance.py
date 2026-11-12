"""Input/code fingerprints for future runs and honest reconstruction of historical ones."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from mlrdegg.paths import PROJECT_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(paths, root=PROJECT_ROOT):
    result = {}
    for path in sorted(paths):
        path = Path(path)
        key = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        result[key] = sha256(path)
    return result


def run_provenance(cfg, ds, cli):
    code = sorted((PROJECT_ROOT / "code").rglob("*.py"))
    code += sorted((PROJECT_ROOT / "code" / "configs").glob("*.yaml"))
    inputs = Path(cfg["preprocess"]["out_dir"])
    if not inputs.is_absolute():
        inputs = PROJECT_ROOT / inputs
    files = [inputs / "space.npz"] + [inputs / f"{s.subject}.npz" for s in ds.subjects]
    effective = json.loads(json.dumps(cfg))
    effective["experiment"].pop("cache_provenance", None)
    payload = {"effective_config": effective, "subjects": [s.subject for s in ds.subjects],
               "samples": {s.subject: len(s.y) for s in ds.subjects},
               "input_sha256": {p.name: sha256(p) for p in files}, "input_directory": str(inputs),
               "code_sha256": fingerprint(code)}
    payload["python"] = platform.python_version()
    payload["dependencies"] = {name: importlib.metadata.version(name) for name in
                               ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "threadpoolctl", "PyYAML")}
    # Parallel worker count changes execution capacity, not the scientific fit.
    # Retain it in the receipt while allowing a memory-constrained run to resume.
    cache_payload = json.loads(json.dumps(payload))
    cache_payload["effective_config"]["experiment"].pop("n_jobs", None)
    payload["cache_fingerprint"] = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()
    payload["cli"] = cli
    try:
        payload["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        payload["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        source = PROJECT_ROOT / "source_provenance.json"
        saved = json.loads(source.read_text(encoding="utf-8")) if source.exists() else {}
        payload["git_commit"] = saved.get("git_commit")
        payload["git_dirty"] = saved.get("git_dirty")
        payload["source_record"] = "portable export" if saved else "Git metadata unavailable"
    return payload
