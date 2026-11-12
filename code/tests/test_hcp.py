"""HCP preprocessing building blocks: mesh geometry, graphs, smoothing and temporal cleaning."""

from pathlib import Path

import numpy as np
import pytest

from mlrdegg.datasets import hcp, hcp_surface
from mlrdegg.experiment.models import build_graphs
from mlrdegg.graphs import mesh_graph
from mlrdegg.paths import resolve

# Unit square split into two triangles, plus a detached third triangle.
COORDS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [5, 5, 0], [6, 5, 0], [5, 6, 0]], float)
TRIS = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])


def test_mesh_edges_unique_and_sorted():
    e = hcp_surface.mesh_edges(TRIS)
    assert e.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3], [4, 5], [4, 6], [5, 6]]


def test_vertex_areas_sum_to_surface_area():
    a = hcp_surface.vertex_areas(COORDS, TRIS)
    assert a.sum() == pytest.approx(1.0 + 0.5)
    assert a[0] == pytest.approx(1.0 / 3.0)  # 1/3 of each of its two triangles (0.5 each)


def test_corrected_edge_lengths_scale_with_sqrt_area_ratio():
    e = hcp_surface.mesh_edges(TRIS)
    area = hcp_surface.vertex_areas(COORDS, TRIS)
    raw = hcp_surface.corrected_edge_lengths(COORDS, e, area, area)
    np.testing.assert_allclose(raw, np.linalg.norm(COORDS[e[:, 0]] - COORDS[e[:, 1]], axis=1))
    np.testing.assert_allclose(hcp_surface.corrected_edge_lengths(COORDS, e, area, 4 * area), 2 * raw)


def test_mesh_graph_weights_and_components():
    e = hcp_surface.mesh_edges(TRIS)
    d = np.linalg.norm(COORDS[e[:, 0]] - COORDS[e[:, 1]], axis=1)
    g = mesh_graph(len(COORDS), e[::-1, ::-1], d[::-1], delta=1.0)  # unsorted, reversed input
    assert g.edges.tolist() == e.tolist()
    np.testing.assert_allclose(g.weights, np.exp(-(d**2) / 2.0))
    assert g.n_components() == 2


def test_build_graphs_mesh_type():
    e = hcp_surface.mesh_edges(TRIS)
    space = {"coords": COORDS, "edges": e, "edge_dist": np.full(len(e), 6.0)}
    sg, tg = build_graphs({"spatial": {"type": "mesh", "delta": 6.0}, "temporal": {"type": "chain"}},
                          space, n_time=4)
    assert sg.n_edges == len(e) and np.allclose(sg.weights, np.exp(-0.5))
    assert tg.n_edges == 3
    sg, tg = build_graphs({"spatial": {"type": "mesh", "delta": 6.0}, "temporal": {"type": "chain"}},
                          space, n_time=1)
    assert tg is None


def test_smooth_volume_keeps_constants_inside_mask():
    mask = np.zeros((12, 12, 12), bool)
    mask[3:9, 2:10, 4:8] = True
    data = np.where(mask[..., None], 5.0, 99.0).astype(np.float32).repeat(3, axis=3)
    out = hcp.smooth_volume(data, mask, fwhm_mm=6.0)
    np.testing.assert_allclose(out[mask], 5.0, rtol=1e-5)  # no leakage from outside the mask
    assert np.all(out[~mask] == 0.0)


def test_smooth_volume_fwhm():
    mask = np.ones((41, 41, 41), bool)
    data = np.zeros((41, 41, 41, 1), np.float32)
    data[20, 20, 20, 0] = 1.0
    prof = hcp.smooth_volume(data, mask, fwhm_mm=8.0)[:, 20, 20, 0]
    x = (np.arange(41) - 20) * hcp.VOXEL_MM
    half = x[prof >= prof.max() / 2]
    assert half.max() - half.min() == pytest.approx(8.0, abs=2 * hcp.VOXEL_MM)


def test_clean_timeseries_removes_drift_and_standardizes():
    n = 400
    t = np.arange(n) * hcp.TR
    rng = np.random.default_rng(0)
    signal = np.sin(2 * np.pi * t / 20.0)
    ts = np.vstack([signal + 3.0 * t / t[-1] + 7.0, signal + rng.normal(0, 0.1, n)])
    out = hcp.clean_timeseries(ts, {"highpass_cutoff_s": 128.0})
    np.testing.assert_allclose(out.mean(1), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(1), 1.0)
    assert np.corrcoef(out[0], signal)[0, 1] > 0.99  # the linear drift is gone


def _surface_resources():
    res = hcp_surface.SurfaceResources(
        workbench=Path(__import__("os").environ.get("WB_COMMAND", "wb_command")),
        atlas_dir=resolve("data/HCP_YA_2025/standard_mesh_atlases"),
        group_dir=resolve("data/HCP_YA_2025/S1200_group_average_surfaces"))
    try:
        res.check()
    except FileNotFoundError:
        pytest.skip("Connectome Workbench or HCP surface files not available")
    return res


def test_fsaverage4_geometry(tmp_path):
    geo = hcp_surface.build_geometry(_surface_resources(), tmp_path)
    V = len(geo["coords"])
    assert geo["valid_L"].size == geo["valid_R"].size == 2562
    assert V == geo["valid_L"].sum() + geo["valid_R"].sum() and 4500 < V < 5000
    assert np.all(geo["coords"][geo["hemi"] == 0, 0] <= 1.0)  # left hemisphere has x <= 0
    assert np.all(geo["coords"][geo["hemi"] == 1, 0] >= -1.0)
    g = mesh_graph(V, geo["edges"], geo["edge_dist"], delta=6.0)
    assert g.n_components() == 2  # one per hemisphere, none across
    assert np.all(geo["hemi"][geo["edges"][:, 0]] == geo["hemi"][geo["edges"][:, 1]])
    deg = np.bincount(geo["edges"].ravel(), minlength=V)
    assert deg.min() >= 1 and 5.5 < deg.mean() < 6.1
    assert 5.0 < np.median(geo["edge_dist"]) < 7.0  # ~6 mm spacing after the area correction
