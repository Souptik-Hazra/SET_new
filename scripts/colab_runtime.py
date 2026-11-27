"""Colab execution helpers for the canonical SET Pharma pipeline.

The existing research implementation remains in src/. This module adds
runtime diagnostics, stage-level checkpoint validation and Drive artifact sync.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "output" / "colab_run_manifest.json"
DISCLAIMER = "Inventory outcomes are SIMULATED. Stockout labels are not observed events."

INGESTION_ARTIFACTS = (
    "data/processed/scmd_monthly_aggregated.parquet",
    "data/processed/scmd_monthly_series.parquet",
    "output/tables/cohort_inclusion_report.csv",
    "output/tables/monthly_data_summary.csv",
    "output/phase1_diagnostics.json",
)
MODEL_ARTIFACTS = (
    "data/processed/features.parquet",
    "data/processed/forecasts.parquet",
    "data/processed/reliability.parquet",
    "data/processed/reliability_baselines.parquet",
    "data/processed/scenario_streams.parquet",
    "data/processed/radcsr_ablations.parquet",
    "data/processed/calibrated_risk_probabilities.parquet",
    "data/processed/raw_risk_probabilities.parquet",
    "data/processed/simulated_inventory.parquet",
    "data/processed/radcsr_risk_details.parquet",
    "output/resolved_config.json",
    "output/radcsr_monotonicity_audit.json",
    "output/pipeline_report.json",
    "output/tables/cohort_inclusion_report.csv",
    "output/tables/monthly_data_summary.csv",
    "output/tables/forecast_provenance.csv",
    "output/tables/forecast_source_summary.csv",
    "output/tables/forecast_metrics.csv",
    "output/tables/per_series_forecast_metrics.csv",
    "output/tables/per_series_outlier_summary.csv",
    "output/tables/reliability_baseline_metrics.csv",
    "output/tables/reliability_signal_correlations.csv",
    "output/tables/observation_counts.csv",
    "output/tables/bootstrap_observation_counts.csv",
    "output/tables/risk_brier_by_scenario.csv",
    "output/tables/risk_calibration_metrics.csv",
    "output/tables/phase7_ablation_metrics.csv",
    "output/tables/phase6_inventory_scenarios.csv",
    "output/tables/radcsr_vs_baselines_bootstrap_holm.csv",
    "output/tables/s5_bias_radcsr_vs_static_wasserstein.csv",
    "output/tables/clinical_priority_ranking_comparison.csv",
    "output/final_verdict.md",
)
MATH_ARTIFACTS = (
    "output/mathematical_validation.json",
    "output/mathematical_validation.md",
)


def assert_supported_python() -> dict:
    """Require the repository's configured Python 3.13 runtime."""
    actual = (sys.version_info.major, sys.version_info.minor)
    if actual != (3, 13):
        raise RuntimeError(
            f"Unsupported Python {sys.version.split()[0]}. "
            "This project requires Python 3.13.x."
        )
    return {"python_version": sys.version.split()[0], "python_supported": True,
            "python_override": False}


def environment_report() -> dict:
    """Return runtime diagnostics without assuming a GPU is available."""
    python_status = assert_supported_python()
    report = {
        "python": sys.version,
        **python_status,
        "platform": platform.platform(),
        "cpu_count": __import__("os").cpu_count(),
    }
    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        report["available_ram_gb"] = round(psutil.virtual_memory().available / (1024 ** 3), 2)
        report["disk_free_gb"] = round(shutil.disk_usage("/content" if Path("/content").exists() else ROOT).free / (1024 ** 3), 2)
    except ImportError:
        report["resource_details"] = "Install psutil for RAM and disk diagnostics."
    try:
        import torch
        report["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch_cuda_available"] = False
    return report


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_digest(paths: tuple[str, ...] | list[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint dependency is missing: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def run_fingerprint(
    source_csv: str | Path,
    parameters: dict | None = None,
    code_paths: tuple[str, ...] | list[str] = (),
) -> str:
    """Fingerprint source metadata, relevant code, config and stage parameters."""
    source = Path(source_csv).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Pipeline input not found: {source}")
    config = ROOT / "src" / "set_forecasting" / "config.py"
    payload = {
        "source": {
            "name": source.name,
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
        },
        "config_sha256": _sha256(config) if config.is_file() else None,
        "code_sha256": _code_digest(code_paths) if code_paths else None,
        "parameters": parameters or {},
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        return {"version": 1, "stages": {}}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read Colab checkpoint manifest: {MANIFEST_PATH}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("stages", {}), dict):
        raise ValueError("Colab checkpoint manifest has an invalid structure.")
    return data


def _write_manifest(data: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(MANIFEST_PATH)


def stage_is_complete(stage: str, fingerprint: str, artifacts: tuple[str, ...]) -> bool:
    """A stage is reusable only when its fingerprint and every output match."""
    entry = _read_manifest()["stages"].get(stage, {})
    if entry.get("fingerprint") != fingerprint:
        return False
    return all((ROOT / artifact).is_file() and (ROOT / artifact).stat().st_size > 0
               for artifact in artifacts)


def run_stage(
    stage: str,
    fingerprint: str,
    artifacts: tuple[str, ...],
    action: Callable[[], object],
    *,
    force: bool = False,
    invalidate_stages: tuple[str, ...] = (),
):
    """Run one stage and mark it complete only after all expected outputs exist."""
    if not force and stage_is_complete(stage, fingerprint, artifacts):
        print(f"[CHECKPOINT] {stage}: matching outputs found; skipping.")
        return {"stage": stage, "status": "SKIPPED_VALID_CHECKPOINT"}

    # Invalidate any previous success record before work begins. If a forced
    # rerun fails, stale outputs must not make the old checkpoint reusable.
    manifest = _read_manifest()
    manifest.setdefault("version", 1)
    stages = manifest.setdefault("stages", {})
    stages.pop(stage, None)
    for downstream_stage in invalidate_stages:
        stages.pop(downstream_stage, None)
    _write_manifest(manifest)

    print(f"[STAGE START] {stage}")
    result = action()
    missing = [item for item in artifacts
               if not (ROOT / item).is_file() or (ROOT / item).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{stage} completed without required artifacts: {missing}")

    manifest = _read_manifest()
    manifest.setdefault("version", 1)
    manifest.setdefault("stages", {})[stage] = {
        "fingerprint": fingerprint,
        "artifacts": list(artifacts),
        "status": "completed",
        "git_revision": _git_revision(),
    }
    _write_manifest(manifest)
    print(f"[STAGE COMPLETE] {stage}")
    return {"stage": stage, "status": "COMPLETED", "result": result}


def ingestion_fingerprint(
    source_csv: str | Path, min_periods: int, min_nonzero: int,
    train_frac: float = 0.60, val_frac: float = 0.20, test_frac: float = 0.20,
) -> str:
    return run_fingerprint(
        source_csv,
        {"stage": "ingestion", "min_periods": min_periods, "min_nonzero": min_nonzero,
         "train_frac": train_frac, "val_frac": val_frac, "test_frac": test_frac},
        code_paths=(
            "src/set_forecasting/data/ingestion.py",
            "src/set_forecasting/data/source.py",
            "src/set_forecasting/config.py",
            "scripts/colab_runtime.py",
        ),
    )


def model_fingerprint(bootstrap: int, seed: int) -> str:
    series_path = ROOT / "data" / "processed" / "scmd_monthly_series.parquet"
    if not series_path.is_file():
        raise FileNotFoundError("Monthly series checkpoint is missing. Run ingestion first.")
    params = {
        "stage": "model_pipeline",
        "series_size": series_path.stat().st_size,
        "series_mtime_ns": series_path.stat().st_mtime_ns,
        "bootstrap": int(bootstrap),
        "seed": int(seed),
    }
    model_code = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src" / "set_forecasting").rglob("*.py")
        if "validation" not in path.relative_to(ROOT).parts
    ]
    return run_fingerprint(
        series_path, params,
        code_paths=model_code + ["src/set_forecasting/config.py", "scripts/colab_runtime.py"],
    )


def require_stage(stage: str, fingerprint: str, artifacts: tuple[str, ...]) -> None:
    if not stage_is_complete(stage, fingerprint, artifacts):
        raise RuntimeError(
            f"Required stage '{stage}' is not complete for the current inputs/code. "
            "Run the preceding notebook stage successfully; do not rely on leftover files."
        )


def run_ingestion(
    source_csv: str | Path,
    *,
    min_periods: int = 18,
    min_nonzero: int = 6,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    force: bool = False,
):
    assert_supported_python()
    from src.set_forecasting.data.ingestion import run_phase1_from_csv

    fingerprint = ingestion_fingerprint(source_csv, min_periods, min_nonzero, train_frac, val_frac, test_frac)
    return run_stage(
        "ingestion", fingerprint, INGESTION_ARTIFACTS,
        lambda: run_phase1_from_csv(
            source=str(Path(source_csv).resolve()),
            min_periods=min_periods,
            min_nonzero=min_nonzero,
            train_frac=train_frac,
            val_frac=val_frac,
            test_frac=test_frac,
            load_series=False,
        ),
        force=force,
        invalidate_stages=("model_pipeline", "mathematical_validation"),
    )


def run_models(
    source_csv: str | Path,
    *,
    min_periods: int = 18,
    min_nonzero: int = 6,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    bootstrap: int = 2000,
    seed: int = 42,
    force: bool = False,
):
    assert_supported_python()
    from src.set_forecasting.pipeline import run_model_pipeline

    ingestion_fp = ingestion_fingerprint(source_csv, min_periods, min_nonzero, train_frac, val_frac, test_frac)
    require_stage("ingestion", ingestion_fp, INGESTION_ARTIFACTS)
    fingerprint = model_fingerprint(bootstrap, seed)
    return run_stage(
        "model_pipeline", fingerprint, MODEL_ARTIFACTS,
        lambda: run_model_pipeline(n_boot=bootstrap, seed=seed),
        force=force,
        invalidate_stages=("mathematical_validation",),
    )


def run_mathematical_validation(
    *,
    bootstrap: int = 2000,
    seed: int = 42,
    force: bool = False,
):
    assert_supported_python()
    from src.set_forecasting.validation.mathematical import run

    model_fp = model_fingerprint(bootstrap, seed)
    require_stage("model_pipeline", model_fp, MODEL_ARTIFACTS)
    series_path = ROOT / "data" / "processed" / "scmd_monthly_series.parquet"
    forecasts_path = ROOT / "data" / "processed" / "forecasts.parquet"
    params = {
        "stage": "mathematical_validation",
        "series_mtime_ns": series_path.stat().st_mtime_ns,
        "forecasts_size": forecasts_path.stat().st_size,
        "forecasts_mtime_ns": forecasts_path.stat().st_mtime_ns,
        "model_fingerprint": model_fp,
    }
    fingerprint = run_fingerprint(
        series_path, params,
        code_paths=(
            "src/set_forecasting/validation/mathematical.py",
            "src/set_forecasting/risk/radcsr.py",
            "src/set_forecasting/reliability/index.py",
            "scripts/colab_runtime.py",
        ),
    )
    return run_stage(
        "mathematical_validation", fingerprint, MATH_ARTIFACTS,
        run, force=force,
    )


def sync_artifacts(source_root: str | Path, destination_root: str | Path) -> list[str]:
    """Copy only processed data and reports; never copy raw data or the repository."""
    source = Path(source_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    copied = []
    for relative in ("data/processed", "output"):
        src = source / relative
        if not src.exists():
            continue
        dst = destination / relative
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            copied.append(str(Path(relative) / rel))
    return copied


print(DISCLAIMER)
