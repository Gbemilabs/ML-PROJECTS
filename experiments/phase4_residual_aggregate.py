from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def summarize(frame: pd.DataFrame, prediction: str) -> dict:
    error = frame[prediction].to_numpy() - frame["y_log"].to_numpy()
    fold_scores = frame.assign(error=error).groupby(["seed", "fold"]).error.apply(
        lambda values: float(np.mean(values.to_numpy() ** 2) ** 0.5)
    )
    return {
        "pooled_rmse": float(np.mean(error**2) ** 0.5),
        "mean_fold_rmse": float(fold_scores.mean()),
        "fold_rmse_std": float(fold_scores.std(ddof=0)),
        "worst_fold_rmse": float(fold_scores.max()),
    }


def main() -> None:
    first = pd.read_csv(EXPERIMENTS / "nested_residual_oof.csv")
    second = pd.read_csv(EXPERIMENTS / "nested_residual_confirm_oof.csv")
    third = pd.read_csv(EXPERIMENTS / "nested_residual_confirm3_oof.csv")
    for frame in (second, third):
        if "corrected_log" in frame.columns:
            frame["gradient_boosting_final_log"] = frame["corrected_log"]
    combined = pd.concat(
        [first.loc[first.seed.eq(2026)], second, third],
        ignore_index=True,
    )
    expected_seeds = {2026, 2027, 2028, 2029, 2030}
    if set(combined.seed.unique()) != expected_seeds:
        raise ValueError("Expected five fixed nested confirmation seeds")

    report = {
        "experiment_id": "P4-RESIDUAL-AGGREGATE",
        "seeds": sorted(expected_seeds),
        "outer_splits": 5,
        "base": summarize(combined, "xgb_base_log"),
        "corrected": summarize(combined, "gradient_boosting_final_log"),
        "by_seed": {},
        "pooled_gain_base_minus_corrected": None,
        "decision": "UNASSESSED",
        "oof_sources": [
            "experiments/nested_residual_oof.csv",
            "experiments/nested_residual_confirm_oof.csv",
            "experiments/nested_residual_confirm3_oof.csv",
        ],
        "external_champion_score": 0.12654,
        "external_score": None,
        "champion_oof_comparison": False,
    }
    for seed, group in combined.groupby("seed"):
        report["by_seed"][str(int(seed))] = {
            "base": summarize(group, "xgb_base_log"),
            "corrected": summarize(group, "gradient_boosting_final_log"),
        }
    report["pooled_gain_base_minus_corrected"] = (
        report["base"]["pooled_rmse"] - report["corrected"]["pooled_rmse"]
    )
    worst_fold_change = (
        report["corrected"]["worst_fold_rmse"]
        - report["base"]["worst_fold_rmse"]
    )
    report["worst_fold_change_corrected_minus_base"] = worst_fold_change
    report["decision"] = (
        "PROMISING_LOCAL_ONLY"
        if report["pooled_gain_base_minus_corrected"] >= 0.0003
        and worst_fold_change <= 0.01
        else "REJECT"
    )
    output = EXPERIMENTS / "nested_residual_combined_validation.json"
    output.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    log_row = {
        "experiment_id": report["experiment_id"],
        "hypothesis": "Fixed residual correction should improve when independent confirmation seeds are pooled.",
        "change": "Aggregate predeclared nested confirmation seeds 2026-2030",
        "cv": f"{report['corrected']['pooled_rmse']:.6f}",
        "cv_std": f"{report['corrected']['fold_rmse_std']:.6f}",
        "validation_scheme": "Nested KFold5 outer / KFold4 inner; five confirmation seeds pooled",
        "runtime_seconds": "",
        "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Across five fixed confirmation seeds, corrected pooled RMSE {report['corrected']['pooled_rmse']:.6f} vs base {report['base']['pooled_rmse']:.6f}; gain={report['pooled_gain_base_minus_corrected']:.6f}; worst-fold change={worst_fold_change:+.6f}. XGBoost-only comparison; champion ensemble nested OOF still required.",
                "status": report["decision"],
    }
    existing_index = next(
        (i for i, row in enumerate(rows) if row.get("experiment_id") == report["experiment_id"]),
        None,
    )
    if existing_index is None:
        rows.append(log_row)
    else:
        rows[existing_index] = log_row
    with log_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()