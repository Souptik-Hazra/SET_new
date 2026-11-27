# Google Colab execution

## Purpose

`notebooks/SET_Pharma_Colab_Pipeline.ipynb` is an execution interface for the existing SET Pharma Python package. Forecasting, feature engineering, reliability, RADCSR, calibration, simulation and mathematical validation remain implemented in `src/`; the notebook does not maintain duplicate model code.

## Setup

1. Push/check out `feature/colab-pipeline-execution`.
2. Upload the authorised SCMD source CSV to Google Drive at:
   `MyDrive/SET_Pharma/data/raw/combined_secondary_care_medicines.csv`
   or edit `DRIVE_RAW_CSV` in the notebook.
3. Open the notebook in Google Colab and run cells in order.
4. Keep the default research parameters unless running a documented experiment.
5. Review the generated JSON/Markdown reports and persist the outputs in Drive.

The raw CSV is copied from Drive to Colab local storage for the ingestion run. Raw data, processed Parquet files, reports and model artifacts must not be committed to Git.

## Checkpoint behavior

The helper in `scripts/colab_runtime.py` records a stage manifest only after all declared outputs exist. A checkpoint is reused only if its fingerprint matches source metadata, hashes of the stage's relevant Python files, the central Python configuration module and stage parameters, and all required artifacts are present. The Git revision is recorded for audit but documentation-only commits do not invalidate compute checkpoints.

The notebook requires a valid ingestion checkpoint before model execution and a valid model checkpoint before mathematical validation. A downstream stage cannot run merely because old output files happen to exist.

The initial workflow has three checkpoint boundaries:

- **Ingestion:** DuckDB aggregation and monthly-series construction.
- **Model pipeline:** existing `run_model_pipeline()` call.
- **Mathematical validation:** Phase B validation and reports.

The current model pipeline is monolithic. If the runtime disconnects while it is executing, the model stage must be rerun; a completed ingestion checkpoint is retained. The notebook does not claim mid-model resume support. A future change may split the model pipeline into independently testable stages, preserving the current outputs and research semantics.

Processed data and reports are copied to Drive after each successful stage. On reconnect, the notebook restores those artifacts before checking the manifest. Google Drive is persistent storage; `/content` is temporary runtime storage.

## Resource guidance

- RAM, CPU and free disk are important for this dataset.
- The source CSV is copied to local Colab disk to avoid repeated Drive reads during DuckDB ingestion.
- A GPU is optional; do not assume it accelerates DuckDB, pandas or all forecasting work.
- The notebook checks available disk before copying the source.
- Do not reduce bootstrap counts, cohort thresholds, training data or simulation settings merely to make a run finish faster without recording the change as an experiment.

## Outputs

The notebook persists `data/processed/` and `output/` under `MyDrive/SET_Pharma_Colab_Artifacts/`. This includes forecasts, reliability outputs, RADCSR outputs, simulated inventory, pipeline reports and mathematical-validation reports.

**Research limitation:** Inventory outcomes are simulated. Stockout labels are not observed events. Results based on simulated labels must not be presented as validation against observed hospital stockouts.


## Research figures

After mathematical validation, the notebook calls `src.visualization.generate_figures.generate_figures()`. The generator reads the current Parquet/CSV outputs and writes available PNGs to `output/figures/`. It does not substitute hard-coded example values when an artifact is missing.

The notebook displays each generated PNG inline with IPython and then synchronizes the complete `output/` directory to Drive. A machine-readable `output/figures/figure_generation_report.json` lists generated files and any skipped figures with the missing-input reason. Re-run the figure cell after changing outputs to refresh the plots.

Figures based on inventory and stockout labels are explicitly marked as simulated. These must not be described as observed hospital stockout results. Figures are generated only when their required source artifacts exist; a skipped figure is preferable to a fabricated or misleading plot.
