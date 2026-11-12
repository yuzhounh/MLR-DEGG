"""HCP-YA cortical surface data resampled from fs_LR 32k to a standard fsaverage mesh.

Resampling follows the HCP pipelines route with Connectome Workbench: fs_LR 32k -> fsaverage
through the ``fs_LR-deformed_to-fsaverage`` sphere, adaptive barycentric area-weighted
interpolation with the group-average vertex areas, medial wall excluded (``atlasroi``).

Geometry comes from the S1200 group-average MSMAll midthickness resampled to the same mesh.  A
group-average surface is shrunk by the averaging of folds (about 60 % of the individual area), so
edge lengths are rescaled by the square root of the ratio between the average individual vertex
area (``midthickness_va_avg``) and the vertex area of the group surface.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

MESH_SIZES = {"fsaverage4": "3k", "fsaverage5": "10k"}
HEMIS = (("L", "CORTEX_LEFT"), ("R", "CORTEX_RIGHT"))


@dataclass(frozen=True)
class SurfaceResources:
    workbench: Path
    atlas_dir: Path  # HCPpipelines standard_mesh_atlases (with resample_fsaverage/)
    group_dir: Path  # S1200 group-average surfaces
    mesh: str = "fsaverage4"

    def __post_init__(self) -> None:
        if self.mesh not in MESH_SIZES:
            raise ValueError(f"unknown mesh {self.mesh}; choose from {sorted(MESH_SIZES)}")

    @property
    def _rs(self) -> Path:
        return self.atlas_dir / "resample_fsaverage"

    def sphere_in(self, h: str) -> Path:
        return self._rs / f"fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii"

    def sphere_out(self, h: str) -> Path:
        return self._rs / f"{self.mesh}_std_sphere.{h}.{MESH_SIZES[self.mesh]}_fsavg_{h}.surf.gii"

    def area_in(self, h: str) -> Path:
        return self._rs / f"fs_LR.{h}.midthickness_va_avg.32k_fs_LR.shape.gii"

    def area_out(self, h: str) -> Path:
        return self._rs / f"{self.mesh}.{h}.midthickness_va_avg.{MESH_SIZES[self.mesh]}_fsavg_{h}.shape.gii"

    def roi_in(self, h: str) -> Path:
        return self.atlas_dir / f"{h}.atlasroi.32k_fs_LR.shape.gii"

    def group_surface(self, h: str, kind: str = "midthickness") -> Path:
        return self.group_dir / f"S1200.{h}.{kind}_MSMAll.32k_fs_LR.surf.gii"

    def check(self) -> None:
        files = [self.workbench] + [f(h) for h in "LR" for f in (
            self.sphere_in, self.sphere_out, self.area_in, self.area_out, self.roi_in, self.group_surface)]
        missing = [str(p) for p in files if not Path(p).exists()]
        if missing:
            raise FileNotFoundError("missing surface resources:\n  " + "\n  ".join(missing))


def wb(res: SurfaceResources, *args) -> None:
    proc = subprocess.run([str(res.workbench), *map(str, args)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"wb_command {args[0]} failed:\n{proc.stdout}\n{proc.stderr}")


def _gifti_data(path: Path) -> np.ndarray:
    darrays = nib.load(str(path)).darrays
    return darrays[0].data if len(darrays) == 1 else np.column_stack([d.data for d in darrays])


def _metric_resample(res: SurfaceResources, h: str, src: Path, dst: Path, valid_out: Path | None = None):
    extra = ["-valid-roi-out", valid_out] if valid_out is not None else []
    wb(res, "-metric-resample", src, res.sphere_in(h), res.sphere_out(h), "ADAP_BARY_AREA", dst,
       "-area-metrics", res.area_in(h), res.area_out(h), "-current-roi", res.roi_in(h), *extra)


def mesh_edges(triangles: np.ndarray) -> np.ndarray:
    t = np.asarray(triangles, dtype=np.int64)
    e = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [0, 2]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def vertex_areas(coords: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """One third of the area of every incident triangle (the Workbench vertex-area convention)."""
    a, b, c = (coords[triangles[:, k]] for k in range(3))
    tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out = np.zeros(len(coords))
    for k in range(3):
        np.add.at(out, triangles[:, k], tri_area / 3.0)
    return out


def corrected_edge_lengths(coords, edges, area_group, area_target) -> np.ndarray:
    """Edge lengths on ``coords`` rescaled by ``sqrt`` of the target / group area ratio at both ends."""
    i, j = edges.T
    raw = np.linalg.norm(coords[i] - coords[j], axis=1)
    return raw * np.sqrt((area_target[i] + area_target[j]) / (area_group[i] + area_group[j]))


def build_geometry(res: SurfaceResources, work_dir: Path) -> dict:
    """Vertices kept on the target mesh (both hemispheres, medial wall excluded) and their graph.

    Returns arrays for ``space.npz``: ``coords`` (group midthickness, mm), ``hemi`` (0 = L),
    ``vertex`` (index on the full target mesh), ``edges`` / ``edge_dist`` in the compact vertex
    order, ``valid_L`` / ``valid_R`` masks and ``triangles_L`` / ``triangles_R`` of the full mesh.
    """
    res.check()
    work_dir.mkdir(parents=True, exist_ok=True)
    out = {"coords": [], "hemi": [], "vertex": [], "edges": [], "edge_dist": []}
    offset = 0
    for k, (h, _) in enumerate(HEMIS):
        valid_path, surf_path = work_dir / f"{h}.valid.shape.gii", work_dir / f"{h}.midthickness.surf.gii"
        _metric_resample(res, h, res.roi_in(h), work_dir / f"{h}.roi.shape.gii", valid_path)
        wb(res, "-surface-resample", res.group_surface(h), res.sphere_in(h), res.sphere_out(h),
           "BARYCENTRIC", surf_path)
        valid = _gifti_data(valid_path) > 0
        coords, tri = (d.data for d in nib.load(str(surf_path)).darrays)
        coords, tri = np.asarray(coords, float), np.asarray(tri, np.int64)
        e = mesh_edges(tri)
        e = e[valid[e].all(axis=1)]
        dist = corrected_edge_lengths(coords, e, vertex_areas(coords, tri), _gifti_data(res.area_out(h)))
        compact = np.full(len(valid), -1)
        compact[valid] = np.arange(valid.sum()) + offset
        out["coords"].append(coords[valid])
        out["hemi"].append(np.full(valid.sum(), k))
        out["vertex"].append(np.flatnonzero(valid))
        out["edges"].append(compact[e])
        out["edge_dist"].append(dist)
        out[f"valid_{h}"], out[f"triangles_{h}"] = valid, tri
        offset += int(valid.sum())
    for key in ("coords", "hemi", "vertex", "edges", "edge_dist"):
        out[key] = np.concatenate(out[key])
    out["mesh"] = np.array(res.mesh)
    return out


def read_run(res: SurfaceResources, dtseries: bytes, work_dir: Path, valid: dict,
             smooth_fwhm_mm: float = 0.0) -> np.ndarray:
    """Cortical time series ``(V, n_frames)`` on the target mesh from one CIFTI dense time series.

    ``smooth_fwhm_mm > 0`` first adds geodesic smoothing on the fs_LR 32k surface (S1200
    midthickness with the va_avg area correction), as the HCP task analyses do.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / "in.dtseries.nii"
    src.write_bytes(dtseries)
    if smooth_fwhm_mm > 0:
        sm = work_dir / "smoothed.dtseries.nii"
        wb(res, "-cifti-smoothing", src, smooth_fwhm_mm, smooth_fwhm_mm, "COLUMN", sm, "-fwhm",
           "-left-surface", res.group_surface("L"), "-left-corrected-areas", res.area_in("L"),
           "-right-surface", res.group_surface("R"), "-right-corrected-areas", res.area_in("R"))
        src = sm
    args = ["-cifti-separate", src, "COLUMN"]
    for h, structure in HEMIS:
        args += ["-metric", structure, work_dir / f"{h}.func.gii"]
    wb(res, *args)
    parts = []
    for h, _ in HEMIS:
        dst = work_dir / f"{h}.target.func.gii"
        _metric_resample(res, h, work_dir / f"{h}.func.gii", dst)
        parts.append(np.asarray(_gifti_data(dst), dtype=np.float32)[valid[h]])
    return np.vstack(parts)
