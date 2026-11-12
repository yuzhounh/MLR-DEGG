# MLR-DEGG

Spatiotemporal Double-Exponent Group-Regularized Multinomial Logistic Regression for Brain Decoding.

Research code accompanying the working manuscript by Xiao Xie, Kunpeng Xia, Yanfang Chen, Hao Cai, Xingliang Xiong, Lihan Wang, Li Zhang, and Jing Wang.

This code publication corresponds to manuscript working version 2.1, including accepted-precision primary HCP Graph-TV solutions and reporting updates dated 12 November 2026. No journal acceptance, article DOI or stable release tag is implied.

## Contents

- code/src/mlrdegg/: identifiable multinomial models, row-group selection, separate spatial/temporal graph penalties, cross-exponent calibration, APG and MM-ADMM solvers, original-objective residual diagnostics, datasets and experiment helpers.
- code/scripts/: preprocessing, simulation, grouped validation, fixed-fit precision refinement, figures and reporting.
- code/configs/: simulation, ERP37 and HCP181 settings. Parallelism defaults to one worker in the public examples.
- code/tests/: synthetic unit and regression tests.

The repository is a source-code publication. It does not redistribute participant data, restricted family metadata, fitted models, coefficients, individual prediction tables, internal reviews, or the unpublished manuscript.

## Setup

Use Python 3.11 or later. From the repository root:

~~~sh
python -m pip install -e ".[neuro,plots,dev]"
python -m pytest code/tests/test_building_blocks.py code/tests/test_primary_refinement.py -q
~~~

The configuration files record scientific settings rather than a historical dependency lockfile. The original HCP181 run used NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3 and scikit-learn 1.6.1; the current code is also checked in the authors' newer environment. Optional solver-benchmark routes additionally need CVXPY and Clarabel.

## Data preparation

ERP CORE is available at [the original data resource](https://doi.org/10.18115/D5JW4R). Obtain the official N170 artifact-parameter files and supply their values in the CSV location configured by preprocess.ar_params. Source data retain their original license and attribution requirements.

HCP Young Adult data and surface resources must be obtained through the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult). Surface preprocessing requires Connectome Workbench: set preprocess.surface.workbench in the HCP preprocessing configurations to its actual executable path. Authorized family metadata are inputs to 20_prepare_hcp_unrelated.py; they are not part of the software repository.

In the HCP181 configurations, subjects: all means all participants in the prepared, family-selected derived directory. It is not an instruction to use every HCP participant. The paper uses 181 unrelated participants, 64 segments each, grouped outer/inner folds, and a separate seven-method family comparison.

## Experiments

The following commands require the appropriately prepared inputs. They start computation; no experiment runs during installation.

~~~sh
python code/scripts/10_simulation_studies.py --help
python code/scripts/03_run_decoding.py --config code/configs/erpcore_n170_37.yaml --dry-run
python code/scripts/03_run_decoding.py --config code/configs/hcp181_primary_tv.yaml --dry-run
python code/scripts/09_selection_rules_report.py --help
~~~

Remove dry-run only when intentionally starting a fit. The HCP181 APG and TV configurations are logical phases that can be run on one machine or separate compute resources. Hardware partition names in precision helpers do not encode network addresses.

## Accepted-precision primary Graph-TV results

The primary comparison keeps the existing grids, folds and selected parameters. Eight distinct selected Graph-TV outer fits supply ten rule/fold rows. All eight reported solutions pass the normalized original-objective common-residual band of 1e-3; seven also met the stricter objective/parameter stopping rules. One retained third-fold iterate met the residual criterion before those stricter rules were met. A residual-band pass is not a claim of global optimality.

The fixed outer-fit continuation protocol uses objective tolerance 1e-8, parameter tolerance 1e-5, a maximum of 5000 outer/1000 inner iterations, CG tolerance 1e-8 and checkpointed accepted outer states. The code preserves signed coefficients, splitting/dual variables, rho and iteration histories for interruption/resume.

26_refine_hcp181_primary.py contains the fixed selected-refit protocol. The two host arguments partition_a and partition_b are logical task partitions, not server addresses. configs/primary_refinement_manifest.json lists the fixed folds/strengths without participant identifiers. Prepare its run inputs and frozen runtime with:

~~~sh
python prepare_precision_run.py --run results/hcp181_primary_tv
python code/scripts/26_refine_hcp181_primary.py --host partition_a --workers 1 --check-only
~~~

The refinement entry point starts computation without check-only. It does not repeat inner selection or half fits. 28_check_primary_checkpoint.py evaluates a retained checkpoint without fitting. These precision helpers intentionally retain the paper-specific run-directory convention.

## Reporting

09_selection_rules_report.py reports a completed decoding run. Figure scripts consume saved result arrays; they do not recover signed coefficients from nonnegative magnitudes.

The paper-specific installers paper_report.py, update_hcp181_report.py and 30_install_primary_refinement.py expect the documented research-workspace result layout and compact report inputs. Those inputs are not bundled here. They are supplied for transparency about the table-generation logic, not as a claim that the paper's tables can be reconstructed without its result products.

Use 15_feature_figures.py --primary-only for the four-member primary HCP display after installing accepted primary results. The separate full-family display keeps its own fit source. Fresh split-half reproducibility, cross-fold magnitude-map stability, and exact nonzero prediction-coefficient rows are different quantities.

## Citation and license

Use CITATION.cff to cite this software. A final article citation can be added when available.

Code is distributed under the MIT license. This license does not replace the access terms or licenses of ERP CORE, HCP, Connectome Workbench, or other external resources.
