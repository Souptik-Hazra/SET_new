"""Generate publication-ready figures from the active SCMD pipeline artifacts.

All empirical plots are built from saved pipeline outputs. Synthetic inventory and
stockout plots are explicitly labelled as simulated. No illustrative metric values
are embedded in this module.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DISCLAIMER = "Inventory outcomes and stockout labels are SIMULATED, not observed hospital events."
KEYS = ["ODS_CODE", "VMP_SNOMED_CODE", "UNIT_OF_MEASURE_IDENTIFIER"]


def _read_parquet(root: Path, relative: str) -> pd.DataFrame | None:
    path = root / relative
    return pd.read_parquet(path) if path.is_file() else None


def _read_csv(root: Path, relative: str) -> pd.DataFrame | None:
    path = root / relative
    return pd.read_csv(path) if path.is_file() else None


def _finish(fig, path: Path, title: str) -> Path:
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _series_sample(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [key for key in KEYS if key in frame.columns]
    if not keys:
        return frame.sort_values("PERIOD").copy()
    counts = frame.groupby(keys, dropna=False).size().sort_values(ascending=False)
    selected = counts.index[0]
    if len(keys) == 1:
        mask = frame[keys[0]].eq(selected)
    else:
        mask = pd.Series(True, index=frame.index)
        for key, value in zip(keys, selected if isinstance(selected, tuple) else (selected,)):
            mask &= frame[key].eq(value)
    return frame.loc[mask].sort_values("PERIOD").copy()


def generate_figures(project_root: str | Path | None = None) -> list[Path]:
    """Create figures from actual pipeline artifacts and return saved paths.

    Missing source artifacts are recorded in output/figures/figure_generation_report.json.
    Figures with simulated outcomes carry an explicit simulated-data label.
    """
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    out = root / "output"
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    skipped: dict[str, str] = {}

    series = _read_parquet(root, "data/processed/scmd_monthly_series.parquet")
    forecasts = _read_parquet(root, "data/processed/forecasts.parquet")
    reliability = _read_parquet(root, "data/processed/reliability.parquet")
    scenarios = _read_parquet(root, "data/processed/scenario_streams.parquet")
    inventory = _read_parquet(root, "data/processed/simulated_inventory.parquet")
    calibrated = _read_parquet(root, "data/processed/calibrated_risk_probabilities.parquet")
    brier = _read_csv(root, "output/tables/risk_brier_by_scenario.csv")
    ablations = _read_csv(root, "output/tables/phase7_ablation_metrics.csv")
    correlations = _read_csv(root, "output/tables/reliability_signal_correlations.csv")
    sensitivity = _read_csv(root, "output/tables/inventory_sensitivity.csv")

    # 1. Observed SCMD quantity distribution: descriptive source-data view, not stock.
    try:
        if series is None or not {"PERIOD", "MODEL_DEMAND"}.issubset(series.columns):
            raise ValueError("monthly series artifact or required columns unavailable")
        frame = series.copy()
        frame["PERIOD"] = pd.to_datetime(frame["PERIOD"])
        frame["MODEL_DEMAND"] = pd.to_numeric(frame["MODEL_DEMAND"], errors="coerce")
        frame = frame[(frame["SPLIT"] == "train") & frame["MODEL_DEMAND"].notna()] if "SPLIT" in frame else frame
        monthly = frame.groupby(frame["PERIOD"].dt.to_period("M").astype(str))["MODEL_DEMAND"].sum()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(monthly.index, monthly.values, marker="o", linewidth=1.5)
        ax.set(title="Monthly SCMD quantity in the training cohort", xlabel="Month", ylabel="Recorded quantity (VMP units)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig01_monthly_scmd_quantity.png", "Figure 1"))
    except Exception as exc:
        skipped["fig01_monthly_scmd_quantity.png"] = str(exc)

    # 2. Test-split actual quantity against the model median forecast for one real series.
    try:
        required = set(KEYS + ["PERIOD", "MODEL_DEMAND", "Q50", "SPLIT"])
        if forecasts is None or not required.issubset(forecasts.columns):
            raise ValueError("forecast artifact missing required columns")
        frame = _series_sample(forecasts[forecasts["SPLIT"] == "test"].copy())
        if frame.empty:
            raise ValueError("no held-out test forecasts")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(frame["PERIOD"], frame["MODEL_DEMAND"], marker="o", label="Actual recorded quantity")
        ax.plot(frame["PERIOD"], frame["Q50"], marker="o", label="Median forecast (Q50)")
        ax.set(xlabel="Month", ylabel="Quantity (VMP units)")
        ax.legend()
        ax.grid(alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig02_test_actual_vs_forecast.png", "Figure 2 — Held-out test series"))
    except Exception as exc:
        skipped["fig02_test_actual_vs_forecast.png"] = str(exc)

    # 3. Test-split quantile interval for the same highest-volume available series.
    try:
        required = set(KEYS + ["PERIOD", "MODEL_DEMAND", "Q10", "Q50", "Q90", "SPLIT"])
        if forecasts is None or not required.issubset(forecasts.columns):
            raise ValueError("forecast artifact missing Q10/Q50/Q90 columns")
        frame = _series_sample(forecasts[forecasts["SPLIT"] == "test"].copy())
        if frame.empty:
            raise ValueError("no held-out test forecasts")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.fill_between(frame["PERIOD"], frame["Q10"], frame["Q90"], alpha=.22, label="P10–P90 forecast interval")
        ax.plot(frame["PERIOD"], frame["MODEL_DEMAND"], marker="o", label="Actual")
        ax.plot(frame["PERIOD"], frame["Q50"], linestyle="--", label="Q50")
        ax.set(xlabel="Month", ylabel="Quantity (VMP units)")
        ax.legend()
        ax.grid(alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig03_test_quantile_interval.png", "Figure 3 — Held-out forecast interval"))
    except Exception as exc:
        skipped["fig03_test_quantile_interval.png"] = str(exc)

    # 4. Prequential reliability signals for one actual forecast series.
    try:
        required = set(KEYS + ["PERIOD", "R_E", "R_B", "R_C", "R_D"])
        if reliability is None or not required.issubset(reliability.columns):
            raise ValueError("reliability artifact missing signal columns")
        frame = _series_sample(reliability)
        fig, ax = plt.subplots(figsize=(11, 5))
        for col in ("R_E", "R_B", "R_C", "R_D"):
            ax.plot(frame["PERIOD"], frame[col], marker=".", label=col)
        ax.set(xlabel="Month", ylabel="Unreliability signal (0–1)", ylim=(-.03, 1.03))
        ax.legend(ncol=4)
        ax.grid(alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig04_reliability_signals.png", "Figure 4 — Prequential reliability signals"))
    except Exception as exc:
        skipped["fig04_reliability_signals.png"] = str(exc)

    # 5. Reliability diagram from held-out test predictions and simulated inventory labels.
    try:
        required = {"PERIOD", "SCENARIO_ID", "METHOD", "SHORTAGE_PROBABILITY", "SPLIT"}
        if calibrated is None or inventory is None or not required.issubset(calibrated.columns):
            raise ValueError("calibrated risk probabilities or inventory labels unavailable")
        if "STOCKOUT_LABEL" not in inventory.columns:
            raise ValueError("simulated inventory lacks STOCKOUT_LABEL")
        join_keys = [key for key in KEYS if key in calibrated.columns and key in inventory.columns] + ["PERIOD", "SCENARIO_ID"]
        frame = calibrated.merge(inventory[join_keys + ["STOCKOUT_LABEL"]], on=join_keys, how="inner", validate="many_to_one")
        frame = frame[(frame["SPLIT"] == "test")].dropna(subset=["SHORTAGE_PROBABILITY", "STOCKOUT_LABEL"])
        if frame.empty:
            raise ValueError("no complete held-out test outcomes for calibration plot")
        fig, ax = plt.subplots(figsize=(7, 6))
        for method, group in frame.groupby("METHOD"):
            bins = pd.cut(group["SHORTAGE_PROBABILITY"].clip(0, 1), bins=np.linspace(0, 1, 11), include_lowest=True)
            curve = group.assign(_BIN=bins).groupby("_BIN", observed=True).agg(
                predicted=("SHORTAGE_PROBABILITY", "mean"), observed=("STOCKOUT_LABEL", "mean"), n=("STOCKOUT_LABEL", "size"))
            if len(curve):
                ax.plot(curve["predicted"], curve["observed"], marker="o", label=str(method))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
        ax.set(xlabel="Mean predicted probability", ylabel="Simulated event frequency", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig05_simulated_label_calibration.png", "Figure 5 — Calibration against simulated labels"))
    except Exception as exc:
        skipped["fig05_simulated_label_calibration.png"] = str(exc)

    # 6. Held-out Brier score by scenario and method.
    try:
        required = {"SCENARIO_ID", "METHOD", "BRIER_SCORE"}
        if brier is None or not required.issubset(brier.columns):
            raise ValueError("risk_brier_by_scenario.csv missing required columns")
        frame = brier.copy()
        pivot = frame.pivot_table(index="SCENARIO_ID", columns="METHOD", values="BRIER_SCORE", aggfunc="mean")
        if pivot.empty:
            raise ValueError("Brier score table is empty")
        fig, ax = plt.subplots(figsize=(13, 6))
        pivot.plot(kind="bar", ax=ax)
        ax.set(xlabel="Synthetic stress scenario", ylabel="Brier score (lower is better)")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Method", fontsize=8)
        ax.grid(axis="y", alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig06_brier_by_scenario.png", "Figure 6 — Brier scores on simulated outcomes"))
    except Exception as exc:
        skipped["fig06_brier_by_scenario.png"] = str(exc)

    # 7. Scenario-level RADCSR risk distribution from actual saved risk outputs.
    try:
        risk = _read_parquet(root, "data/processed/radcsr_risk_details.parquet")
        if risk is None or not {"SCENARIO_ID", "SPLIT", "RADCSR_PROBABILITY"}.issubset(risk.columns):
            raise ValueError("RADCSR risk detail artifact missing required columns")
        frame = risk[(risk["SPLIT"] == "test")].copy()
        if frame.empty:
            raise ValueError("no held-out RADCSR risk rows")
        fig, ax = plt.subplots(figsize=(12, 5))
        frame.boxplot(column="RADCSR_PROBABILITY", by="SCENARIO_ID", ax=ax, grid=False, rot=35)
        ax.set(xlabel="Synthetic stress scenario", ylabel="RADCSR risk score")
        fig.suptitle("")
        generated.append(_finish(fig, fig_dir / "fig07_radcsr_risk_by_scenario.png", "Figure 7 — RADCSR risk scores on held-out simulated scenarios"))
    except Exception as exc:
        skipped["fig07_radcsr_risk_by_scenario.png"] = str(exc)

    # 8. Inventory outcome summary, explicitly synthetic.
    try:
        required = {"SCENARIO_ID", "STOCKOUT_LABEL", "SHORTAGE_QUANTITY", "SPLIT"}
        if inventory is None or not required.issubset(inventory.columns):
            raise ValueError("simulated inventory artifact missing required columns")
        if scenarios is None or not {"PERIOD", "SCENARIO_ID", "SPLIT"}.issubset(scenarios.columns):
            raise ValueError("scenario stream split metadata unavailable")
        split_keys = [key for key in KEYS if key in inventory.columns and key in scenarios.columns] + ["PERIOD", "SCENARIO_ID"]
        frame = inventory.merge(scenarios[split_keys + ["SPLIT"]].drop_duplicates(), on=split_keys, how="inner", validate="one_to_one")
        frame = frame[(frame["SPLIT"] == "test")].dropna(subset=["STOCKOUT_LABEL"])
        if frame.empty:
            raise ValueError("no complete held-out simulated inventory horizons")
        summary = frame.groupby("SCENARIO_ID").agg(
            simulated_stockout_rate=("STOCKOUT_LABEL", "mean"),
            mean_shortage_quantity=("SHORTAGE_QUANTITY", "mean"))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        summary["simulated_stockout_rate"].plot(kind="bar", ax=axes[0])
        summary["mean_shortage_quantity"].plot(kind="bar", ax=axes[1])
        axes[0].set_ylabel("Simulated stockout rate")
        axes[1].set_ylabel("Mean shortage quantity")
        axes[1].set_xlabel("Synthetic stress scenario")
        axes[1].tick_params(axis="x", rotation=35)
        for ax in axes:
            ax.grid(axis="y", alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig08_simulated_inventory_outcomes.png", "Figure 8 — Simulated inventory outcomes (not observed events)"))
    except Exception as exc:
        skipped["fig08_simulated_inventory_outcomes.png"] = str(exc)

    # 9. RADCSR component ablation from the current pipeline table.
    try:
        required = {"SCENARIO_ID", "METHOD", "BRIER_SCORE"}
        if ablations is None or not required.issubset(ablations.columns):
            raise ValueError("phase7_ablation_metrics.csv missing required columns")
        pivot = ablations.pivot_table(index="SCENARIO_ID", columns="METHOD", values="BRIER_SCORE", aggfunc="mean")
        if pivot.empty:
            raise ValueError("ablation table is empty")
        fig, ax = plt.subplots(figsize=(12, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set(xlabel="Synthetic stress scenario", ylabel="Brier score")
        ax.tick_params(axis="x", rotation=35)
        ax.legend(title="Ablation", fontsize=8)
        ax.grid(axis="y", alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig09_radcsr_ablation.png", "Figure 9 — RADCSR ablation on simulated outcomes"))
    except Exception as exc:
        skipped["fig09_radcsr_ablation.png"] = str(exc)

    # 10. Empirical correlation matrix of the four saved reliability signals.
    try:
        if correlations is None or correlations.empty:
            raise ValueError("reliability_signal_correlations.csv unavailable")
        matrix = correlations.set_index(correlations.columns[0])
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        if matrix.empty or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError("correlation matrix has insufficient dimensions")
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix.to_numpy(float), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        generated.append(_finish(fig, fig_dir / "fig10_reliability_signal_correlations.png", "Figure 10 — Reliability signal correlations"))
    except Exception as exc:
        skipped["fig10_reliability_signal_correlations.png"] = str(exc)


    # 11. Optional inventory-assumption sensitivity, if the dedicated grid was run.
    try:
        required = {"SERVICE_LEVEL", "INITIAL_COVER_MONTHS", "TARGET_COVER_MONTHS", "SIMULATED_STOCKOUT_RATE"}
        if sensitivity is None or not required.issubset(sensitivity.columns):
            raise ValueError("inventory_sensitivity.csv unavailable; run scripts/run_inventory_sensitivity.py to create it")
        frame = sensitivity.copy()
        frame["POLICY"] = frame.apply(
            lambda row: f"Initial {row['INITIAL_COVER_MONTHS']:g}m / Target {row['TARGET_COVER_MONTHS']:g}m",
            axis=1,
        )
        summary = frame.groupby(["SERVICE_LEVEL", "POLICY"])["SIMULATED_STOCKOUT_RATE"].mean().unstack("POLICY")
        if summary.empty:
            raise ValueError("inventory sensitivity table is empty")
        fig, ax = plt.subplots(figsize=(11, 5))
        summary.plot(marker="o", ax=ax)
        ax.set(xlabel="Assumed service level", ylabel="Mean simulated stockout rate")
        ax.legend(title="Inventory cover policy", fontsize=8)
        ax.grid(alpha=.25)
        generated.append(_finish(fig, fig_dir / "fig11_inventory_sensitivity.png", "Figure 11 — Inventory assumption sensitivity (simulated outcomes)"))
    except Exception as exc:
        skipped["fig11_inventory_sensitivity.png"] = str(exc)

    report = {
        "disclaimer": DISCLAIMER,
        "generated_count": len(generated),
        "generated": [path.name for path in generated],
        "skipped": skipped,
        "note": "Only figures generated from current saved pipeline artifacts are included. Missing inputs are reported, never replaced with fabricated values.",
    }
    (fig_dir / "figure_generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Figures generated: {len(generated)}; skipped: {len(skipped)}")
    for path in generated:
        print(f"[FIGURE] {path}")
    for name, reason in skipped.items():
        print(f"[SKIPPED] {name}: {reason}")
    print(DISCLAIMER)
    return generated


def main() -> None:
    generate_figures()


if __name__ == "__main__":
    main()
