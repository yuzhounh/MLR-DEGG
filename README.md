# MLR-DEGG

Research code for spatiotemporal double-exponent group-regularized multinomial logistic regression for brain decoding.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Paper:** Jing Wang, Xiao Xie, Kunpeng Xia, Yanfang Chen, Hao Cai, Xingliang Xiong, Lihan Wang, and Li Zhang. *Spatiotemporal Double-Exponent Group-Regularized Multinomial Logistic Regression for Brain Decoding*.

## Overview

MLR-DEGG combines row-group feature selection with separate spatial and temporal graph penalties whose exponents and calibrated strengths define a common family of structured decoders (including Group lasso, GraphNet, Graph TV, Temporal graph TV, Superquadratic, and nonconvex variants). The repository provides the core solver package (`src/mlrdegg/`), preprocessing and decoding pipelines for simulated, ERP CORE N170, and Human Connectome Project (HCP) fMRI datasets, solver benchmarks, and result/figure generation scripts.

## Data Access and Setup

No participant neuroimaging data, restricted family metadata, or fitted model checkpoints are distributed in this repository.

- **ERP CORE N170**: Available from the [original resource](https://doi.org/10.18115/D5JW4R) (requires the N170 artifact-parameter file).
- **HCP Young Adult**: Obtain neuroimaging data, surface files, and authorized family metadata through the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult); surface preprocessing uses Connectome Workbench (`preprocess.surface.workbench`).

See [data/README.md](data/README.md) for expected directory layout under `data/`.

## Environment and Usage

Requires Python 3.11 or later. Install the package in editable mode from the repository root:

```bash
python -m pip install -e ".[neuro,plots,dev]"
pytest
```

The recorded HCP181 run used Python 3.12.7, NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, and scikit-learn 1.6.1 (see [requirements.txt](requirements.txt)).

### Running Experiments

- **Simulation studies**: `scripts/10_simulation_studies.py`, `scripts/11_simulation_studies_report.py`, and `scripts/14_simulation_figure.py` (with `configs/simulation_studies.yaml`).
- **ERP and HCP decoding**: `scripts/03_run_decoding.py` with the ERP37 and HCP181 configurations in `configs/`.
- **HCP common-grid fits**: `configs/hcp181_primary_apg.yaml` and `configs/hcp181_primary_tv.yaml` (HCP APG objective tolerance is $10^{-6}$; simulation and ERP use the default $10^{-9}$).

### Replaying Saved Results

When completed local fit outputs are available, the reporting scripts reproduce the supplementary tables and figures without refitting:

```bash
# ERP zero graph strength (Tables S3 and S4)
python scripts/replay_erp_zero.py --positive-run PATH_TO_ERP37_PRIMARY --zero-run PATH_TO_ERP37_ZERO_GRAPH --output-dir reports/erp-zero

# HCP common grid (Table S5 and Figure S3)
python scripts/hcp_common_grid.py --apg-run PATH_TO_ORIGINAL_APG --tv-run PATH_TO_ORIGINAL_TV --output-dir reports/hcp-common
python scripts/15_feature_figures.py --primary-only --primary-dir reports/hcp-common --surface-dir PATH_TO_SURFACE_FILES --dest reports/figures

# HCP expanded grids (Table S6)
python scripts/44_analyze_boundary_results.py --primary-dir reports/hcp-common --refit-manifest PATH_TO_REFIT_MANIFEST.json --refit-input PATH_TO_ORIGINAL_REFIT_OUTPUTS.zip --output-dir reports/hcp-expanded
```

For staging expanded-grid sensitivity runs from completed common-grid outputs, use `python scripts/prepare_boundary_run.py --primary-dir reports/hcp-common --phase <inner|half|refit>`.

## Repository Structure

```text
MLR-DEGG/
├── README.md                  # Overview, setup, and usage
├── LICENSE                    # MIT license
├── CITATION.cff               # Citation metadata
├── pyproject.toml             # Package configuration
├── requirements.txt           # Recorded package versions
├── src/mlrdegg/               # Core estimator, solvers, graphs, datasets, and evaluation library
├── scripts/                   # Experiment, preprocessing, replay, table, and figure scripts
├── configs/                   # YAML experiment configurations
├── tests/                     # Unit tests (pytest)
└── data/
    └── README.md              # External dataset access instructions
```

## License and Contact

Released under the [MIT License](LICENSE). ERP CORE, HCP, and Connectome Workbench retain their respective licenses and data-use terms.

Correspondence: Jing Wang (`wangjing@xynu.edu.cn`).
