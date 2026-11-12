"""Dataset preprocessing and loading.

Every preprocessed dataset is stored under ``data/derived/<name>/`` as

* ``space.npz`` - spatial layout shared by all subjects (coordinates, names, ...);
* ``sub-<id>.npz`` - trials ``X`` of shape ``(n, V, T)`` with labels and grouping variables.
"""

from .common import DerivedDataset, load_derived

__all__ = ["DerivedDataset", "load_derived"]
