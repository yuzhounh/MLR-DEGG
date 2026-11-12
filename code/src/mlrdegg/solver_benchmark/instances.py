"""Small simulated instances with frozen calibration and stored initial values (plan Section 3).

Data reuse the four-class "spatial template x temporal profile + correlated Gaussian noise"
mechanism of ``code/scripts/05_simulation_benchmark.py`` with the 32 x 32 / six-point shapes of
``simulation_shapes.yaml``.  Templates are area/interval averaged onto the small grids (so thin
shapes do not vanish) and every class template is rescaled to a fixed Frobenius norm.  The noise
smoothing widths are rescaled with the grid so that the normalized correlation length is kept.

Calibration and convex warm starts call the existing estimator code (reference-radius pilot,
``build_branches`` and the estimator's own warm-start solve) once per instance; all routes then
read the stored branches and initial value.
"""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter

from .. import likelihood as lik
from ..contrast import contrast_matrix
from ..estimator import MLRDEGGClassifier
from ..graphs import chain_graph, gaussian_graph, vectorize
from ..penalties import row_norms
from ..solver import SolverState
from .core import Problem, branch_arrays, load_instance, plain, save_npz, write_json

POINTS = {  # (p_s, p_g); p_g None: no active graph (C0 deletes the graph term)
    "C0": (1.0, None), "C1": (1.0, 1.0), "C2": (1.0, 1.5), "C3": (1.0, 2.0), "C4": (1.0, 3.0),
    "C5": (1.5, 2.0), "N1": (0.5, 2.0), "N2": (1.0, 0.5), "N3": (0.5, 0.5),
}


def _load_shape_patterns(project_root: Path):
    """``shape_patterns`` of the existing simulation script (reused, not reimplemented)."""
    scripts = project_root / "code" / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("_sim05", scripts / "05_simulation_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.shape_patterns


def overlap_matrix(n_fine: int, n_coarse: int) -> np.ndarray:
    """Averaging weights: coarse cell i covers [i s, (i+1) s), fine cell j covers [j, j+1)."""
    s = n_fine / n_coarse
    R = np.zeros((n_coarse, n_fine))
    for i in range(n_coarse):
        lo, hi = i * s, (i + 1) * s
        for j in range(n_fine):
            R[i, j] = max(0.0, min(hi, j + 1.0) - max(lo, float(j))) / s
    return R


def scaled_templates(shapes_cfg: dict, grid, n_time: int, n_classes: int, fro_norm: float,
                     shape_patterns) -> np.ndarray:
    """Class templates ``(K, V, T)`` on the small grid, each with Frobenius norm ``fro_norm``."""
    src = shapes_cfg["simulation"]
    fine_grid, fine_t = tuple(src["grid"]), int(src["n_time"])
    shapes = src["shapes"][:n_classes]
    fine = shape_patterns(fine_grid, shapes)
    Rr, Rc = overlap_matrix(fine_grid[0], grid[0]), overlap_matrix(fine_grid[1], grid[1])
    Rt = overlap_matrix(fine_t, n_time)
    out = []
    for k, sh in enumerate(shapes):
        spatial = Rr @ fine[k].reshape(fine_grid) @ Rc.T
        profile = Rt @ np.asarray(sh.get("profile", src["temporal_profile"]), dtype=float)
        P = np.outer(spatial.ravel(), profile)
        norm = np.linalg.norm(P)
        if norm == 0:
            raise ValueError(f"template {sh['name']} vanished on grid {grid}")
        out.append(P * (fro_norm / norm))
    return np.stack(out)


def simulate(seed: int, templates: np.ndarray, grid, sizes: dict, s_sigma: float, t_sigma: float):
    """Same draw as ``05_simulation_benchmark.simulate`` with the rescaled templates."""
    rng = np.random.default_rng(seed)
    K, V, T = templates.shape

    def draw(n):
        y = np.arange(n) % K
        rng.shuffle(y)
        noise = rng.normal(size=(n, T) + tuple(grid))
        if s_sigma > 0 or t_sigma > 0:
            noise = gaussian_filter(noise, sigma=(0, t_sigma) + (s_sigma,) * len(grid))
            noise /= noise.std()
        Z = np.moveaxis(noise.reshape(n, T, V), 1, 2) + templates[y]
        return vectorize(Z), y

    return draw(sizes["n_train"]), draw(sizes["n_val"]), draw(sizes["n_test"])


def grid_graphs(grid, n_time: int, graph_cfg: dict):
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in grid], indexing="ij"), -1).reshape(-1, 2)
    coords = np.column_stack([coords, np.zeros(len(coords))]).astype(float)
    s = graph_cfg["spatial"]
    spatial = gaussian_graph(coords, s["delta"], s["kappa"])
    temporal = chain_graph(n_time, graph_cfg.get("temporal", {}).get("weight", 1.0))
    return spatial, temporal


def _hash_arrays(arrays: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def environment() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "processor": platform.processor()}


def instance_id(scale: str, seed: int, point: str, strength: str) -> str:
    return f"{scale}_s{seed}_{point}_{strength}"


def build_dataset(cfg: dict, scale: str, seed: int, project_root: Path) -> dict:
    """Training/validation/test data, standardization and graphs of one (scale, seed)."""
    import yaml

    sim = cfg["simulation"]
    sc = cfg["scales"][scale]
    grid, T = tuple(sc["grid"]), int(sc["n_time"])
    shapes_cfg = yaml.safe_load(open(project_root / sim["shapes_config"], encoding="utf-8"))
    src = shapes_cfg["simulation"]
    templates = scaled_templates(shapes_cfg, grid, T, sim["n_classes"], sim["template_fro_norm"],
                                 _load_shape_patterns(project_root))
    s_sigma = float(src["noise_smoothing"]) * grid[0] / src["grid"][0]
    t_sigma = float(src.get("noise_time_smoothing", 0.0)) * T / src["n_time"]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = simulate(seed, templates, grid, sc, s_sigma, t_sigma)
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    z = lambda X: (X - mean) / std  # noqa: E731  (training-set parameters for every split)
    spatial, temporal = grid_graphs(grid, T, cfg["graph"])
    return {"grid": grid, "n_time": T, "templates": templates, "s_sigma": s_sigma,
            "t_sigma": t_sigma, "Xtr": z(Xtr), "ytr": ytr, "Xva": z(Xva), "yva": yva,
            "Xte": z(Xte), "yte": yte, "mean": mean, "std": std, "spatial": spatial,
            "temporal": temporal}


def build_instance(cfg: dict, data: dict, scale: str, seed: int, point: str, strength: str,
                   directory: Path) -> dict:
    """Frozen calibration, branches and initial value; writes ``<id>.npz`` and ``<id>.json``."""
    p_s, p_g = POINTS[point]
    lam_sel, lam_g = cfg["strengths"][strength]
    if p_g is None:
        lam_g, p_g_model = 0.0, 2.0  # group-lasso convention: graph term deleted
    else:
        p_g_model = p_g
    V = int(np.prod(data["grid"]))
    T = data["n_time"]
    d = V * T
    K = int(cfg["simulation"]["n_classes"])
    solver_cfg = dict(cfg["frozen_estimator"])
    clf = MLRDEGGClassifier(n_space=V, n_time=T, spatial_graph=data["spatial"],
                            temporal_graph=data["temporal"], lambda_sel=lam_sel, lambda_g=lam_g,
                            eta=cfg["graph"]["eta"], p_s=p_s, p_g=p_g_model, **solver_cfg)
    Y = np.eye(K)[data["ytr"]]
    H = contrast_matrix(K)
    Xa = lik.augment(data["Xtr"])
    prior = np.bincount(data["ytr"], minlength=K) / len(data["ytr"])
    logp = np.log(np.maximum(prior, 1e-12))
    theta_null = np.zeros((d + 1, K - 1))
    theta_null[-1] = H.T @ (logp - logp.mean())
    _, G0, _ = lik.loss_grad(Xa, Y, theta_null, H)
    clf.lambda_max_ = float(d * np.max(row_norms(G0[:-1])))
    t0 = time.perf_counter()
    # Reference radii of every branch (also used to convert smoothing amounts); the estimator's
    # calibration uses them only where it needs them (p != 1), exactly as in fit().
    smoothing_radii = clf._reference_radii(Xa, Y, H, theta_null, G0)
    radius_source = dict(clf.reference_radius_source_)
    needs_ref = p_s != 1.0 or (lam_g > 0 and p_g_model != 1.0)
    clf.t_ref_ = dict(smoothing_radii) if needs_ref else {}
    clf.reference_radius_source_ = radius_source if needs_ref else {}
    branches = clf.build_branches()
    calibration_seconds = time.perf_counter() - t0

    nonconvex = p_s < 1 or (lam_g > 0 and p_g_model < 1)
    warm = {"used": False}
    theta0 = theta_null.copy()
    if nonconvex:
        wb = clf.build_branches(max(p_s, 1.0), max(p_g_model, 1.0))
        t0 = time.perf_counter()
        res0 = clf._solve(Xa, Y, H, wb, clf._options(), SolverState(Theta=theta_null.copy()))
        warm = {"used": True, "exponents": [max(p_s, 1.0), max(p_g_model, 1.0)],
                "solver": res0.solver, "status": res0.status, "converged": bool(res0.converged),
                "n_outer": int(res0.n_outer), "seconds": time.perf_counter() - t0,
                "objective_convex_member": float(res0.objective)}
        theta0 = res0.Theta.copy()

    iid = instance_id(scale, seed, point, strength)
    arrays = {"Xa": Xa, "Y": Y, "H": H, "Xva": lik.augment(data["Xva"]), "yva": data["yva"],
              "Xte": lik.augment(data["Xte"]), "yte": data["yte"], "theta0": theta0,
              "theta_null": theta_null, **branch_arrays(branches)}
    directory.mkdir(parents=True, exist_ok=True)
    save_npz(directory / f"{iid}.npz", **arrays)
    data_hash = _hash_arrays({k: arrays[k] for k in ("Xa", "Y", "Xva", "yva", "Xte", "yte")})
    meta = {
        "id": iid, "scale": scale, "seed": seed, "point": point, "strength": strength,
        "p_s": p_s, "p_g": p_g, "lambda_sel": lam_sel, "lambda_g": lam_g,
        "eta": cfg["graph"]["eta"], "nonconvex": bool(nonconvex),
        "n_train": int(Xa.shape[0]), "n_val": int(len(data["yva"])), "n_test": int(len(data["yte"])),
        "grid": list(data["grid"]), "n_time": T, "d": d, "n_classes": K,
        "lambda_max": clf.lambda_max_, "calibration_radii": clf.t_ref_,
        "smoothing_radii": smoothing_radii, "radius_source": radius_source,
        "calibration_seconds": calibration_seconds,
        "branches": [{"name": br.name, "p": br.p, "eps": br.eps, "n_rows": int(br.mu.size),
                      "mu_sum": float(br.mu.sum()), "mu_min": float(br.mu.min()),
                      "mu_max": float(br.mu.max())} for br in branches],
        "initial": "convex_warm_start" if nonconvex else "class_frequency_intercept_zero_coefficients",
        "warm_start": warm,
        "noise_smoothing": {"space": data["s_sigma"], "time": data["t_sigma"]},
        "template_fro_norm": cfg["simulation"]["template_fro_norm"],
        "data_sha256": data_hash, "standardization": "training mean/std (ddof=0)",
        "estimator_settings": solver_cfg, "environment": environment(),
    }
    problem = Problem(load_instance_from_arrays(arrays, branches, meta))
    meta["objective_theta0"] = problem.objective(theta0)
    meta["objective_theta_null"] = problem.objective(theta_null)
    meta["lip_loss"] = problem.lip_loss
    meta["weak_bound"] = problem.weak_bound()
    meta["npz_sha256"] = hashlib.sha256((directory / f"{iid}.npz").read_bytes()).hexdigest()
    write_json(directory / f"{iid}.json", meta)
    return meta


def load_instance_from_arrays(arrays: dict, branches, meta: dict) -> dict:
    inst = {k: arrays[k] for k in ("Xa", "Y", "H", "Xva", "yva", "Xte", "yte", "theta0",
                                   "theta_null")}
    inst["branches"] = branches
    inst["meta"] = meta
    return inst


def planned_instances(cfg: dict) -> list[tuple[str, int, str, str]]:
    """(scale, seed, point, strength) of phases A, B and C (plan Section 7)."""
    out = []
    seen = set()

    def add(*key):
        if key not in seen:
            seen.add(key)
            out.append(key)

    a = cfg["phases"]["A"]
    for point in POINTS:
        add("A", a["seed"], point, "S1")
    for point in a["extra_strength_points"]:
        add("A", a["seed"], point, "S2")
    b = cfg["phases"]["B"]
    for seed in b["seeds"]:
        for strength in b["strengths"]:
            for point in b["points"]:
                add("B", seed, point, strength)
    for point in b["apg_overlap_points"]:
        add("B", b["apg_overlap_seed"], point, b["apg_overlap_strength"])
    c = cfg["phases"]["C"]
    for point in c["points"]:
        add("C", c["seed"], point, c["strength"])
    return out


def prepare_instances(cfg: dict, directory: Path, project_root: Path, log=print) -> list[dict]:
    """Build (or reuse) every planned instance; returns their metadata."""
    metas, datasets = [], {}
    for scale, seed, point, strength in planned_instances(cfg):
        iid = instance_id(scale, seed, point, strength)
        meta_path = directory / f"{iid}.json"
        if meta_path.exists() and (directory / f"{iid}.npz").exists():
            metas.append(load_instance(directory / f"{iid}.npz")["meta"])
            continue
        key = (scale, seed)
        if key not in datasets:
            datasets[key] = build_dataset(cfg, scale, seed, project_root)
        t0 = time.perf_counter()
        meta = build_instance(cfg, datasets[key], scale, seed, point, strength, directory)
        log(f"instance {iid}: d={meta['d']}, n={meta['n_train']}, warm start "
            f"{meta['warm_start'].get('status', 'not used')}, {time.perf_counter() - t0:.2f} s")
        metas.append(meta)
    write_json(directory / "instances.json", plain(metas))
    return metas
