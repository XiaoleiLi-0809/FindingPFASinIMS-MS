# Code for Submission

This folder collects the latest source code used for the PFAS riMp1 manuscript analyses, figures, SRM 2585 prediction, and desktop screening app.

Standalone scripts were renamed by function for readability. Local imports in the copied files were patched so the renamed helper modules can still be imported. The original project files were not modified.

The large trained RF-MC model bundles are not duplicated here. They remain in `../models/rf_sklearn_bundle_POS.joblib` and `../models/rf_sklearn_bundle_NEG.joblib`.

Most scripts expect this folder to remain directly inside the project root so that `Path(__file__).resolve().parent.parent` points to the data, model, output, manuscript, and SI files.

## Contents

### SRM2585
- `srm_audit_tc_separated_isotope_pairing.py`: Audit confirming equal-mass/coeluting but TC-separated features remain independent M features.
- `srm_build_masterlist_match_workbook.mjs`: Build the SRM 2585 master-list mass-match workbook.
- `srm_match_top_pfas_like_peaks_to_master_list.py`: Mass-match SRM 2585 top PFAS-like peaks to the PFAS master list.
- `srm_predict_srm2585_after_first_isotopic_peak_ratio_rename.py`: Historical SRM 2585 prediction audit after the first_isotopic_peak_ratio rename.
- `srm_predict_srm2585_with_tc_separated_experimental_riMp1.py`: Apply the RF-MC model to SRM 2585 using TC-separated experimental riMp1 envelopes.

### app
- `app_launch_pfas_screening_gui.py`: Launcher for the PFAS screening GUI.
- `app_readme.md`: English README for the PFAS screening app.
- `app_readme_zh_CN.md`: Chinese README for the PFAS screening app.
- `pfas_screening_app/__init__.py`: PFAS screening app package source; original module name preserved for imports.
- `pfas_screening_app/__main__.py`: PFAS screening app package source; original module name preserved for imports.
- `pfas_screening_app/engine.py`: PFAS screening app package source; original module name preserved for imports.
- `pfas_screening_app/gui.py`: PFAS screening app package source; original module name preserved for imports.

### baseline
- `baseline_analyze_linear_planes_and_generate_interactive_figure2.py`: Linear baseline plane metrics and interactive 3D Figure 2 generation.

### core
- `core_calculate_experimental_riMp1_isotope_envelope.py`: Final experimental isotope-envelope and riMp1 calculation workflow.

### document
- `document_update_feature_engineering_justification.py`: Update the manuscript feature-engineering justification text.
- `document_update_si_final_model_results.py`: Update SI model-performance tables and figures from final model outputs.
- `document_update_validation_tables.py`: Update validation tables in the manuscript/SI documents.

### environment
- `environment_build_desktop_app.ps1`: PowerShell build entry point for the PFAS screening app.
- `environment_build_portable_desktop_app.ps1`: PowerShell build workflow for the portable Windows app.
- `environment_pyinstaller_pfas_screening_app.spec`: PyInstaller specification for the desktop app.
- `environment_pyinstaller_sitecustomize.py`: Runtime sitecustomize helper used during app packaging.
- `environment_python_requirements.txt`: Python package requirements for the app and analysis scripts.

### figure
- `figure_draw_figure1_chemical_library_2d_filters.py`: Generate Figure 1 chemical-library mz-CCS and riMp1-CCS 2D filter panels.
- `figure_draw_figure3_two_step_linear_filters.py`: Generate Figure 3 two-step linear-filter panels.
- `figure_draw_figure5_1_svm_legacy_filter_overlap.py`: Generate the SVM legacy mz-CCS/riMp1-CCS overlap figure.
- `figure_draw_mc_probability_uncertainty_validation_test.py`: Generate MC probability-uncertainty validation/test plots.
- `figure_draw_roc_prc_overlay_pos_neg.py`: Generate combined POS/NEG ROC and PRC overlays.
- `figure_generate_interactive_3d_heuristic_filter_html.py`: Generate rotatable HTML for the legacy 3D heuristic-filter visualization.
- `figure_render_html_downloaded_3d_views.py`: Use downloaded HTML screenshots to assemble the 3D Figure 2 panels.
- `figure_update_final_model_manuscript_figures.py`: Update manuscript figure files from final model outputs.

### interpretation
- `interpretation_analyze_riMp2_mixed_halogen_exclusion.py`: riMp2 mixed-halogen exclusion-risk analysis supporting the decision not to use riMp2.
- `interpretation_compute_fno_bin_recognition_by_method.py`: F_No-bin recognition analysis across baseline and RF-MC methods.
- `interpretation_compute_mass_stratified_internal_shap.py`: Final mass-stratified internal SHAP visualization workflow.
- `interpretation_mc_shap_permutation_helper.py`: Helper functions for MC-aware SHAP/permutation-style model explanations.
- `interpretation_run_pos_neg_pca_diagnostics.py`: POS/NEG PCA diagnostics for the final feature space.

### model
- `model_external_validation_and_test_rf_mc.py`: External validation/test application of saved RF-MC bundles.
- `model_recompute_fdr_thresholds_and_compare_tiers.py`: Recompute validation-selected FDR thresholds and compare tier outputs.
- `model_shared_rf_mc_data_loading_and_metrics.py`: Shared data loading, labeling, metrics, and self-contained RF helpers.
- `model_train_validate_test_rf_mc_models.py`: Official sklearn-based train/validation/test RF-MC workflow.

### submission
- `06_submission_utilities/assemble_code_for_submission_folder.py`: Utility used to assemble this function-named code folder.
- `submission_prepare_full_package_with_figures_models_tables.py`: Assemble the broader submission package containing documents, figures, models, and tables.

### test
- `tests/test_pfas_screening_engine.py`: Unit tests for the PFAS screening engine and TC-separated isotope pairing.

## Manifest

`manifest_code_for_submission.csv` records the original source file, copied file name, SHA-256 checksum, and source modification time for each file.

## Quick Checks

From the project root, a syntax-only check can be run with:

```powershell
& 'C:\Users\Xiaolei\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m py_compile code_for_submission\*.py
```

The GUI package source is kept under `pfas_screening_app/` with its standard module names to preserve Python package imports.
