from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
MIN_GROUP_SIZE = 20
MIN_SUPPORTED_FOLD_SIZE = 5


def aggregate_group(
    name: str,
    values: pd.Series,
    frame: pd.DataFrame,
    output: list[dict],
) -> None:
    keys = values.fillna("Missing").astype(str).reset_index(drop=True)
    for label, indices in keys.groupby(keys, sort=False).groups.items():
        rows = np.asarray(list(indices), dtype=int)
        if len(rows) < MIN_GROUP_SIZE:
            continue
        log_error = frame.loc[rows, "log_error"].to_numpy()
        dollar_error = frame.loc[rows, "dollar_error"].to_numpy()
        relative_error = frame.loc[rows, "relative_error"].to_numpy()
        fold_means = []
        for fold in range(5):
            fold_rows = rows[frame.loc[rows, "fold"].to_numpy() == fold]
            if len(fold_rows) >= MIN_SUPPORTED_FOLD_SIZE:
                fold_means.append(float(frame.loc[fold_rows, "log_error"].mean()))
        output.append(
            {
                "slice": name,
                "group": str(label),
                "n": len(rows),
                "log_bias_actual_minus_pred": float(log_error.mean()),
                "log_rmse": float(np.mean(log_error * log_error) ** 0.5),
                "supported_fold_count": len(fold_means),
                "log_bias_sd_across_supported_folds": float(np.std(fold_means)) if fold_means else np.nan,
                "dollar_bias_pred_minus_actual": float(dollar_error.mean()),
                "dollar_rmse": float(np.mean(dollar_error * dollar_error) ** 0.5),
                "relative_bias_pred_over_actual": float(relative_error.mean()),
                "mean_abs_relative_error": float(np.mean(np.abs(relative_error))),
            }
        )


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    if champion_oof.Id.duplicated().any() or set(champion_oof.Id) != set(train.Id):
        raise ValueError("Champion OOF must contain each training Id exactly once")
    champion_oof = champion_oof.set_index("Id").reindex(train.Id).reset_index()

    actual_price = train.SalePrice.to_numpy(dtype=float)
    predicted_log = champion_oof.candidate_oof_log.to_numpy(dtype=float)
    predicted_price = np.expm1(predicted_log)
    actual_log = np.log1p(actual_price)
    frame = train.copy()
    frame["log_error"] = actual_log - predicted_log
    frame["dollar_error"] = predicted_price - actual_price
    frame["relative_error"] = predicted_price / actual_price - 1.0
    fold = np.empty(len(train), dtype=int)
    for fold_id, (_, valid) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(train)):
        fold[valid] = fold_id
    frame["fold"] = fold

    total_sf = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
    area_bins = pd.qcut(train.GrLivArea, q=5, duplicates="drop")
    total_bins = pd.qcut(total_sf, q=5, duplicates="drop")
    age = train.YrSold - train.YearBuilt
    age_bins = pd.qcut(age, q=5, duplicates="drop")
    remodel_age = train.YrSold - train.YearRemodAdd
    remodel_bins = pd.qcut(remodel_age, q=5, duplicates="drop")
    year_bins = pd.cut(
        train.YearBuilt,
        bins=[1800, 1945, 1970, 1990, 2000, 2007, 2011],
        include_lowest=True,
    )
    slices = [
        ("Neighborhood", train.Neighborhood),
        ("OverallQual", train.OverallQual),
        ("OverallCond", train.OverallCond),
        ("GrLivArea_quintile", area_bins),
        ("TotalSF_quintile", total_bins),
        ("Age_quintile", age_bins),
        ("RemodelAge_quintile", remodel_bins),
        ("YearBuilt_era", year_bins),
        ("GarageCars", train.GarageCars),
        ("SaleType", train.SaleType),
        ("SaleCondition", train.SaleCondition),
        ("KitchenQual", train.KitchenQual),
        (
            "Neighborhood_x_Quality",
            train.Neighborhood.astype(str) + "|Q" + train.OverallQual.astype(str),
        ),
        (
            "Neighborhood_x_Size",
            train.Neighborhood.astype(str) + "|" + area_bins.astype(str),
        ),
        (
            "Neighborhood_x_SaleCondition",
            train.Neighborhood.astype(str) + "|" + train.SaleCondition.astype(str),
        ),
        (
            "Quality_x_Size",
            train.OverallQual.astype(str) + "|" + area_bins.astype(str),
        ),
        (
            "Quality_x_Age",
            train.OverallQual.astype(str) + "|" + age_bins.astype(str),
        ),
        (
            "Quality_x_RemodelAge",
            train.OverallQual.astype(str) + "|" + remodel_bins.astype(str),
        ),
        (
            "SaleType_x_Quality",
            train.SaleType.astype(str) + "|" + train.OverallQual.astype(str),
        ),
    ]
    output: list[dict] = []
    for name, values in slices:
        aggregate_group(name, values, frame, output)
    errors = frame.log_error.to_numpy()
    output_path = EXPERIMENTS / "phase6_champion_error_map.csv"
    report_path = EXPERIMENTS / "phase6_champion_error_map.json"
    if output_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite phase-six champion error-map artifacts")
    result = pd.DataFrame(output).sort_values(["log_rmse", "n"], ascending=[False, False])
    result.to_csv(output_path, index=False)
    report = {
        "experiment_id": "P6-CHAMPION-ERROR-MAP",
        "prediction_source": "experiments/champion/oof_0.12374_crossfit.csv; candidate OOF averaged over five nested correction seeds",
        "rows": len(train),
        "pooled_log_rmse": float(np.mean(errors * errors) ** 0.5),
        "mean_log_error": float(errors.mean()),
        "dollar_rmse": float(np.mean(frame.dollar_error**2) ** 0.5),
        "mean_absolute_relative_error": float(np.mean(np.abs(frame.relative_error))),
        "slices": len(result),
        "largest_error_slices": result.head(20).to_dict(orient="records"),
        "outputs": [str(output_path.relative_to(ROOT)), str(report_path.relative_to(ROOT))],
        "caveat": "Crossfit errors are descriptive; candidate group corrections require new nested validation.",
    }
    report_path.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": "The new champion residuals reveal stable error families beyond the Phase 5 support-pooled correction.",
                "change": "Log/dollar/relative OOF slices over 19 domain features and interactions",
                "cv": f"{report['pooled_log_rmse']:.6f}",
                "cv_std": "",
                "validation_scheme": "Champion crossfit OOF; descriptive subgroup fold stability",
                "runtime_seconds": "",
                "oof_correlation": "",
                "decision": "DIAGNOSTIC",
                "reason": f"Saved {len(result)} slices; identify only repeatable residual families for next experiment",
                "status": "DIAGNOSTIC",
            }
        )
    print(pd.Series({k: v for k, v in report.items() if k != "largest_error_slices"}).to_string())
    print("Largest error slices:\n", result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()