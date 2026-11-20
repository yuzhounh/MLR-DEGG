"""HCP Young Adult working-memory task fMRI: stimulus-category decoding from block segments.

Input: ``<subject>_Task3TRecommended.zip`` packages with the ICA-cleaned runs of both phase
encodings (TR = 0.72 s): the MNI 2 mm volumes ``tfMRI_WM_{LR,RL}_hp0_clean_rclean_tclean.nii.gz``
with their brain masks, the MSMAll CIFTI dense time series
``tfMRI_WM_{LR,RL}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii`` and the block EV files.
Each run has eight 27.5 s blocks (0-back/2-back x body/faces/places/tools).

Pipeline per run, in three layers:

1. Spatial (:class:`VolumeSpace` or :class:`SurfaceSpace`) -> ``(V, n_frames)``.  Volume:
   optional Gaussian smoothing inside the mask, then block-averaging to a coarse isotropic grid
   inside the intersection of all brain masks.  Surface: optional geodesic smoothing, then
   resampling of the cortex to a standard fsaverage mesh (:mod:`.hcp_surface`).
2. Temporal: DCT high-pass detrending and per-location z-scoring over the run.
3. Samples (``samples: segments``): each block is shifted by a hemodynamic delay and cut into
   non-overlapping segments of ``segment_trs`` frames, averaged in bins of ``bin_trs`` frames.
   One segment is one ``V x T`` sample.  With ``samples: glm`` the temporal layer is replaced by
   a first-level GLM (:func:`glm_maps`) and every z map is one ``V x 1`` sample.
"""

from __future__ import annotations

import gzip
import logging
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter

from . import hcp_glm, hcp_surface
from .common import save_subject

log = logging.getLogger(__name__)

TR = 0.72
VOXEL_MM = 2.0
RUNS = ("LR", "RL")
CATEGORIES = ("body", "faces", "places", "tools")
LOADS = ("0bk", "2bk")
FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))


def zip_path(raw_root: Path, subject: str) -> Path:
    return raw_root / f"{subject}_Task3TRecommended.zip"


def _member(subject: str, run: str, name: str) -> str:
    return f"{subject}/MNINonLinear/Results/tfMRI_WM_{run}/{name}"


def _read_nifti(zf: zipfile.ZipFile, member: str) -> nib.Nifti1Image:
    return nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(member)))


def read_mask(raw_root: Path, subject: str, run: str) -> tuple[np.ndarray, np.ndarray]:
    with zipfile.ZipFile(zip_path(raw_root, subject)) as zf:
        img = _read_nifti(zf, _member(subject, run, "brainmask_fs.2.nii.gz"))
    return np.asarray(img.dataobj) > 0, img.affine


def read_blocks(zf: zipfile.ZipFile, subject: str, run: str) -> list[dict]:
    blocks = []
    for load in LOADS:
        for cat in CATEGORIES:
            text = zf.read(_member(subject, run, f"EVs/{load}_{cat}.txt")).decode()
            for line in text.strip().splitlines():
                onset, duration, _ = map(float, line.split())
                blocks.append({"onset": onset, "duration": duration, "category": cat, "load": load})
    return sorted(blocks, key=lambda b: b["onset"])


def coarse_grid(mask: np.ndarray, affine: np.ndarray, factor: int, min_fraction: float):
    """Coarse voxels (``factor``^3 blocks) whose fraction inside ``mask`` is at least ``min_fraction``."""
    pad = [(0, (-s) % factor) for s in mask.shape]
    m = np.pad(mask.astype(float), pad)
    sh = [s // factor for s in m.shape]
    frac = m.reshape(sh[0], factor, sh[1], factor, sh[2], factor).mean(axis=(1, 3, 5))
    keep = frac >= min_fraction
    ijk = np.argwhere(keep)
    centers_fine = ijk * factor + (factor - 1) / 2.0
    coords = nib.affines.apply_affine(affine, centers_fine)
    return keep, ijk, coords, sh


def downsample(data: np.ndarray, mask: np.ndarray, keep: np.ndarray, factor: int) -> np.ndarray:
    """Mask-weighted block average of a 4-D run -> ``(V, n_frames)``."""
    pad = [(0, (-s) % factor) for s in mask.shape]
    m = np.pad(mask.astype(np.float32), pad)
    sh = [s // factor for s in m.shape]
    count = m.reshape(sh[0], factor, sh[1], factor, sh[2], factor).sum(axis=(1, 3, 5))
    out = np.empty((int(keep.sum()), data.shape[3]), dtype=np.float32)
    for t in range(data.shape[3]):
        vol = np.pad(data[..., t], pad) * m
        s = vol.reshape(sh[0], factor, sh[1], factor, sh[2], factor).sum(axis=(1, 3, 5))
        out[:, t] = (s / np.maximum(count, 1.0))[keep]
    return out


def smooth_volume(data: np.ndarray, mask: np.ndarray, fwhm_mm: float) -> np.ndarray:
    """Gaussian smoothing of every frame restricted to ``mask`` (normalized convolution), so that
    no signal from outside the brain leaks in."""
    sigma = fwhm_mm * FWHM_TO_SIGMA / VOXEL_MM
    m = mask.astype(np.float32)
    norm = gaussian_filter(m, sigma)
    out = np.zeros_like(data, dtype=np.float32)
    inside = norm > 1e-6
    for t in range(data.shape[3]):
        s = gaussian_filter(data[..., t] * m, sigma)
        out[..., t][inside] = s[inside] / norm[inside]
    return out * m[..., None]


class VolumeSpace:
    """MNI 2 mm volumes block-averaged to a ``2 * factor`` mm grid inside the group brain mask."""

    def __init__(self, cfg: dict):
        self.factor = int(cfg["downsample_factor"])
        self.min_fraction = cfg["min_mask_fraction"]
        self.smooth_fwhm_mm = float(cfg.get("smooth_fwhm_mm", 0.0))

    def prepare(self, raw_root: Path, subjects: list[str], work_dir: Path, n_jobs: int = 1) -> dict:
        """Intersection of all run masks (all runs are in the same MNI 2 mm space)."""
        masks = Parallel(n_jobs=n_jobs)(
            delayed(read_mask)(raw_root, s, r) for s in subjects for r in RUNS
        )
        affine = masks[0][1]
        if any(not np.allclose(a, affine) for _, a in masks):
            raise ValueError("run masks are not in a common space")
        self.group_mask = np.logical_and.reduce([m for m, _ in masks])
        self.keep, ijk, coords, coarse_shape = coarse_grid(self.group_mask, affine, self.factor,
                                                           self.min_fraction)
        log.info("group mask %d voxels -> %d coarse voxels (%d mm)", self.group_mask.sum(), len(ijk),
                 int(VOXEL_MM * self.factor))
        return {"coords": coords, "ijk": ijk, "coarse_shape": np.array(coarse_shape),
                "affine": affine, "factor": np.array(self.factor)}

    def read_run(self, zf: zipfile.ZipFile, subject: str, run: str, work_dir: Path) -> np.ndarray:
        img = _read_nifti(zf, _member(subject, run, f"tfMRI_WM_{run}_hp0_clean_rclean_tclean.nii.gz"))
        data = np.asarray(img.dataobj, dtype=np.float32)
        del img
        if self.smooth_fwhm_mm > 0:
            data = smooth_volume(data, self.group_mask, self.smooth_fwhm_mm)
        return downsample(data, self.group_mask, self.keep, self.factor).astype(float)


class SurfaceSpace:
    """Cortex (both hemispheres, no medial wall) of the MSMAll CIFTI data on an fsaverage mesh."""

    def __init__(self, cfg: dict, resolve=Path):
        s = cfg["surface"]
        self.res = hcp_surface.SurfaceResources(
            workbench=resolve(s["workbench"]), atlas_dir=resolve(s["atlas_dir"]),
            group_dir=resolve(s["group_dir"]), mesh=s.get("mesh", "fsaverage4"))
        self.smooth_fwhm_mm = float(cfg.get("smooth_fwhm_mm", 0.0))

    def prepare(self, raw_root: Path, subjects: list[str], work_dir: Path, n_jobs: int = 1) -> dict:
        geo = hcp_surface.build_geometry(self.res, work_dir / "geometry")
        self.valid = {"L": geo["valid_L"], "R": geo["valid_R"]}
        log.info("%s cortex: %d + %d vertices, %d edges, median corrected edge %.2f mm",
                 self.res.mesh, geo["valid_L"].sum(), geo["valid_R"].sum(), len(geo["edges"]),
                 np.median(geo["edge_dist"]))
        return geo

    def read_run(self, zf: zipfile.ZipFile, subject: str, run: str, work_dir: Path) -> np.ndarray:
        member = _member(subject, run, f"tfMRI_WM_{run}_Atlas_MSMAll_hp0_clean_rclean_tclean.dtseries.nii")
        with tempfile.TemporaryDirectory(dir=work_dir, prefix=f"{subject}_{run}_") as tmp:
            return hcp_surface.read_run(self.res, zf.read(member), Path(tmp), self.valid,
                                        self.smooth_fwhm_mm).astype(float)


def make_space(cfg: dict, resolve=Path):
    kind = cfg.get("space", "volume")
    if kind == "volume":
        return VolumeSpace(cfg)
    if kind == "surface":
        return SurfaceSpace(cfg, resolve)
    raise ValueError(f"unknown space {kind}")


def dct_highpass(ts: np.ndarray, tr: float, cutoff_s: float) -> np.ndarray:
    """Remove a constant and a DCT drift basis with periods longer than ``cutoff_s`` (SPM style)."""
    n = ts.shape[1]
    k = int(np.floor(2.0 * n * tr / cutoff_s))
    t = np.arange(n)
    basis = [np.ones(n)] + [np.cos(np.pi * j * (2 * t + 1) / (2 * n)) for j in range(1, k + 1)]
    Q, _ = np.linalg.qr(np.column_stack(basis))
    return ts - (ts @ Q) @ Q.T


def clean_timeseries(ts: np.ndarray, cfg: dict) -> np.ndarray:
    ts = dct_highpass(ts, TR, cfg["highpass_cutoff_s"])
    return (ts - ts.mean(1, keepdims=True)) / np.maximum(ts.std(1, keepdims=True), 1e-8)


def segment_blocks(ts: np.ndarray, blocks: list[dict], cfg: dict) -> list[dict]:
    seg, bin_trs = cfg["segment_trs"], cfg["bin_trs"]
    out = []
    for b_idx, b in enumerate(blocks):
        start = int(np.ceil((b["onset"] + cfg["hrf_delay_s"]) / TR))
        stop = int(np.floor((b["onset"] + b["duration"] + cfg["hrf_delay_s"]) / TR))
        n_seg = min((stop - start) // seg, cfg.get("max_segments_per_block", 99))
        for s in range(n_seg):
            a = start + s * seg
            if a + seg > ts.shape[1]:
                break
            Z = ts[:, a : a + seg].reshape(ts.shape[0], seg // bin_trs, bin_trs).mean(axis=2)
            out.append({"Z": Z, "block": b_idx, "segment": s, **{k: b[k] for k in ("category", "load")}})
    return out


def glm_maps(ts: np.ndarray, blocks: list[dict], cfg: dict) -> list[dict]:
    """z maps (``V x 1`` samples) from a first-level GLM of one run (:mod:`.hcp_glm`).

    ``glm.unit: run`` models the eight load x category EVs and returns one map per category
    (category vs fixation, both loads weighted 1/2, the HCP BODY/FACE/PLACE/TOOL contrasts);
    ``glm.unit: block`` gives every block its own regressor and returns one map per block.
    """
    g = cfg.get("glm", {})
    unit = g.get("unit", "run")
    n = ts.shape[1]
    if unit == "run":
        conds = {f"{l}_{c}": [(b["onset"], b["duration"]) for b in blocks
                              if b["load"] == l and b["category"] == c] for l in LOADS for c in CATEGORIES}
    elif unit == "block":
        conds = {f"b{i}": [(b["onset"], b["duration"])] for i, b in enumerate(blocks)}
    else:
        raise ValueError(f"unknown glm unit {unit}")
    X, names, Q = hcp_glm.design(conds, n, TR, g.get("highpass_cutoff_s", 200.0))
    Y = ts.T - Q @ (Q.T @ ts.T)
    p = X.shape[1]
    if unit == "run":
        C = np.zeros((len(CATEGORIES), p))
        for k, c in enumerate(CATEGORIES):
            for l in LOADS:
                C[k, names.index(f"{l}_{c}")] = 0.5
        info = [{"block": -1, "category": c, "load": "all"} for c in CATEGORIES]
    else:
        C = np.eye(len(names), p)
        info = [{"block": i, "category": b["category"], "load": b["load"]} for i, b in enumerate(blocks)]
    z = hcp_glm.fit_contrasts(Y, X, C, n_drift=Q.shape[1], ar1=g.get("ar1", True))
    return [{"Z": z[k][:, None], "segment": 0, **info[k]} for k in range(len(info))]


def sample_times(cfg: dict) -> np.ndarray:
    if cfg.get("samples", "segments") == "glm":
        return np.zeros(1)
    return (np.arange(cfg["segment_trs"] // cfg["bin_trs"]) + 0.5) * cfg["bin_trs"] * TR


def preprocess_run(space, raw_root: Path, subject: str, run: str, cfg: dict,
                   work_dir: Path) -> list[dict]:
    with zipfile.ZipFile(zip_path(raw_root, subject)) as zf:
        ts = space.read_run(zf, subject, run, work_dir)
        blocks = read_blocks(zf, subject, run)
    if cfg.get("samples", "segments") == "glm":
        segs = glm_maps(ts, blocks, cfg)
    else:
        segs = segment_blocks(clean_timeseries(ts, cfg), blocks, cfg)
    for s in segs:
        s["run"] = run
    log.info("sub-%s run %s: %d segments", subject, run, len(segs))
    return segs


def label_of(seg: dict, label_mode: str) -> str:
    return seg["category"] if label_mode == "category" else f"{seg['load']}_{seg['category']}"


def save_hcp_subject(out_dir: Path, subject: str, segs: list[dict], label_mode: str) -> dict:
    X = np.stack([s["Z"] for s in segs])
    y = np.array([label_of(s, label_mode) for s in segs])
    save_subject(out_dir / f"sub-{subject}.npz", X, y,
                 run=np.array([s["run"] for s in segs]),
                 block=np.array([s["block"] for s in segs]),
                 segment=np.array([s["segment"] for s in segs]),
                 load=np.array([s["load"] for s in segs]))
    return {"subject": subject, "n": len(y), "counts": dict(zip(*np.unique(y, return_counts=True)))}
