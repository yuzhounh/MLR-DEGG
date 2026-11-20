"""Loading of preprocessed (derived) datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SubjectData:
    subject: str
    X: np.ndarray  # (n, V, T)
    y: np.ndarray  # (n,) string labels
    meta: dict  # per-trial arrays (run, block, ...)


@dataclass
class DerivedDataset:
    name: str
    space: dict
    times: np.ndarray
    subjects: list[SubjectData]

    @property
    def n_space(self) -> int:
        return self.subjects[0].X.shape[1]

    @property
    def n_time(self) -> int:
        return self.subjects[0].X.shape[2]


def save_subject(path: Path, X: np.ndarray, y: np.ndarray, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, X=X.astype(np.float32), y=np.asarray(y).astype(str), **meta)
    tmp.replace(path)


def _norm(s) -> str:
    """``"sub-001"``, ``"001"`` and ``1`` denote the same subject."""
    s = str(s).removeprefix("sub-")
    return s.lstrip("0") or "0" if s.isdigit() else s


def load_derived(directory: Path, subjects: list[str] | None = None,
                 exclude: list | None = None) -> DerivedDataset:
    """Load preprocessed subjects, optionally restricted to ``subjects`` and without ``exclude``."""
    directory = Path(directory)
    with np.load(directory / "space.npz", allow_pickle=False) as f:
        space = {k: f[k] for k in f.files}
    files = sorted(directory.glob("sub-*.npz"))
    if exclude:
        dropped = {_norm(s) for s in exclude}
        files = [f for f in files if _norm(f.stem) not in dropped]
    if subjects is not None:
        wanted = {_norm(s) for s in subjects}
        files = [f for f in files if _norm(f.stem) in wanted]
    if not files:
        raise FileNotFoundError(f"no preprocessed subjects in {directory}")
    subs = []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            meta = {k: z[k] for k in z.files if k not in ("X", "y")}
            subs.append(SubjectData(f.stem, z["X"], z["y"], meta))
    return DerivedDataset(directory.name, space, space["times"], subs)
