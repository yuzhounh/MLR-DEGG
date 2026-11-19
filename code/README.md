# MLR-DEGG

Research code for *Spatiotemporal Double-Exponent Group-Regularized Multinomial Logistic Regression for Brain Decoding* by Xiao Xie, Kunpeng Xia, Yanfang Chen, Hao Cai, Xingliang Xiong, Lihan Wang, Li Zhang, and Jing Wang.

This code snapshot accompanies working manuscript v2.4 (T01). The reporting commands below use the original fits returned by each solver at its experimental stopping settings.

## Setup

Use Python 3.11 or later. Run the examples from the repository root. ERP table formatting uses `pdflatex` with the T1, Latin Modern and AMS math packages.

~~~sh
python -m pip install -e ".[neuro,plots,dev]"
~~~

The configuration files record the scientific settings. The recorded HCP181 run used Python 3.12.7, NumPy 2.1.3, SciPy 1.15.3, pandas 2.2.3, and scikit-learn 1.6.1. Software records for some earlier simulation and ERP fits are incomplete.

## Replay original results

These commands read completed local fit outputs and write new report products. Use authorized copies of the original inputs. The code repository contains code and configuration files. Participant data, restricted family metadata, model files, predictions, coefficients, and the unpublished manuscript stay in their authorized locations.

### ERP zero graph strength (Tables S3 and S4)

~~~sh
python replay_erp_zero.py --positive-run PATH_TO_ERP37_PRIMARY --zero-run PATH_TO_ERP37_ZERO_GRAPH --output-dir reports/erp-zero
~~~

The positive-run directory supplies `config.yaml`, `outer_results.csv`, `inner_scores.csv`, `fresh_repro_seed5003.csv`, and `units/*_maps.npz`. The zero-run directory supplies its completed outer and fresh split-half results, maps, `selection.csv`, `new_fit_log.csv`, `completion.json`, and `validation.json`. The replay checks the original selections, repeats the zero-strength selection using the Group lasso objective at zero, and verifies the resulting selections against the saved fits. It writes `erp_zero_tables.tex` and the underlying summary, paired-comparison and zero-selection CSVs. An optional `--manuscript-reference` checks the original ERP table from its aggregate CSV.

### HCP common grid (Table S5 and Figure S3)

~~~sh
python code/scripts/hcp_common_grid.py --apg-run PATH_TO_ORIGINAL_APG --tv-run PATH_TO_ORIGINAL_TV --output-dir reports/hcp-common
python code/scripts/15_feature_figures.py --primary-only --primary-dir reports/hcp-common --surface-dir PATH_TO_SURFACE_FILES --dest reports/figures
~~~

The APG and TV directories each supply `config.yaml`, `inner_scores.csv`, `outer_results.csv`, `units/pooled_predictions.csv`, and `units/pooled_maps.npz`. Use the common-grid directories produced directly by the model runs. The common-grid replay validates the settings, folds, prediction rows, coefficient map shapes and source markers. It writes a source receipt, the merged completed outputs, and `common_grid_detail_rows.tex`. The figure command uses that same merged source and reads `space.npz` and both `inflated.{L,R}.surf.gii` files from the fsaverage4 preprocessing output. It writes `hcp_primary_features.pdf`.

### HCP expanded grids (Table S6)

~~~sh
python code/scripts/44_analyze_boundary_results.py --primary-dir reports/hcp-common --refit-manifest PATH_TO_REFIT_MANIFEST.json --refit-input PATH_TO_ORIGINAL_REFIT_OUTPUTS.zip --output-dir reports/hcp-expanded
~~~

Repeat `--refit-input` for each original worker archive or directory of task subdirectories. Every new refit listed in the manifest must supply its `summary.json`, `predictions.csv`, and `importance.npz`. The analyzer reads archives in place. It verifies the task identifiers, settings, prediction rows and coefficient maps, and uses the same common-grid output for reused points. The standalone replay writes `table_rows.tex` and the summary and test CSVs to the requested output directory.

## Experiments from prepared data

Running experiments requires the appropriate data access, prepared inputs, and original cohort and split settings. ERP CORE is available from the [original data resource](https://doi.org/10.18115/D5JW4R). The ERP preprocessing needs the N170 artifact-parameter file. Obtain HCP Young Adult data and the surface resources through the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult). HCP preprocessing uses Connectome Workbench. Set `preprocess.surface.workbench` to its proper executable. The cohort selection script expects authorized family metadata locally. `subjects: all` refers to the selected derived directory for the relevant HCP configuration. In this study it contains 181 unrelated participants with 64 segments each.

- Simulation main table and maps use `10_simulation_studies.py`, `11_simulation_studies_report.py` and `14_simulation_figure.py`.
- The ERP and HCP main comparisons use `03_run_decoding.py` with the ERP37 and HCP181 family configurations. The reporting scripts consume retained result files and the saved inputs needed by each report. ERP37 uses the original exclusion and split settings. HCP uses participant-grouped folds.
- HCP common-grid fits use `hcp181_primary_apg.yaml` and `hcp181_primary_tv.yaml`. The HCP APG objective tolerance is \(10^{-6}\). Simulation and ERP use the implementation default \(10^{-9}\).

For the expanded HCP experiment, `prepare_boundary_run.py --primary-dir reports/hcp-common --phase inner` prepares a plan. The same source must be passed for `--phase half` and `--phase refit`. Each stage is prepared after the prior stage has completed. The historical worker is Windows-specific and defaults to one worker in the public commands. The refit worker produces a ZIP that the Table S6 analyzer can read directly.

## Citation and license

Use `CITATION.cff` to cite the software. Code is distributed under MIT. ERP CORE, HCP, and Connectome Workbench retain their respective access terms and licenses.
