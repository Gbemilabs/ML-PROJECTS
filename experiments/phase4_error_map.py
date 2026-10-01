from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
MIN_GROUP_SIZE = 12
MIN_FOLD_GROUP_SIZE = 5


def add_slice(rows: list[dict], name: str, values: pd.Series, frame: pd.DataFrame) -> None:
    keys = values.fillna("Missing").astype(str).reset_index(drop=True)
    for label, row_indices in keys.groupby(keys, sort=False).groups.items():
        indices = np.asarray(list(row_indices), dtype=int)
        if len(indices) < MIN_GROUP_SIZE:
            continue
        log_error = frame.loc[indices, "log_error"].to_numpy()
        dollar_error = frame.loc[indices, "dollar_error"].to_numpy()
        relative_error = frame.loc[indices, "relative_error"].to_numpy()
        fold_bias = []
        for fold in range(5):
            fold_indices = indices[frame.loc[indices, "fold"].to_numpy() == fold]
            if len(fold_indices) >= MIN_FOLD_GROUP_SIZE:
                fold_bias.append(float(frame.loc[fold_indices, "log_error"].mean()))
        rows.append(
            {
                "slice": name,
                "group": str(label),
                "n": len(indices),
                "log_bias_actual_minus_pred": float(log_error.mean()),
                "log_rmse": float(np.mean(log_error * log_error) ** 0.5),
                "log_bias_sd_across_supported_folds": float(np.std(fold_bias)) if fold_bias else np.nan,
                "dollar_bias_pred_minus_actual": float(dollar_error.mean()),
                "dollar_rmse": float(np.mean(dollar_error * dollar_error) ** 0.5),
                "relative_bias_pred_over_actual": float(relative_error.mean()),
                "mean_abs_relative_error": float(np.mean(np.abs(relative_error))),
            }
        )


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    oof = pd.read_csv(EXPERIMENTS / "baseline" / "oof_predictions.csv")
    if not np.array_equal(train["Id"].to_numpy(), oof["Id"].to_numpy()):
        raise ValueError("Baseline OOF IDs must align exactly with training data")

    actual_price = train["SalePrice"].to_numpy(dtype=float)
    predicted_price = oof["SalePrice"].to_numpy(dtype=float)
    actual_log = np.log1p(actual_price)
    predicted_log = np.log1p(predicted_price)
    frame = train.copy()
    frame["log_error"] = actual_log - predicted_log
    frame["dollar_error"] = predicted_price - actual_price
    frame["relative_error"] = predicted_price / actual_price - 1.0
    fold_ids = np.empty(len(train), dtype=int)
    for fold, (_, valid_idx) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=42).split(train), start=0
    ):
        fold_ids[valid_idx] = fold
    frame["fold"] = fold_ids

    total_sf = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
    area_bin = pd.qcut(train["GrLivArea"], q=5, duplicates="drop")
    total_area_bin = pd.qcut(total_sf, q=5, duplicates="drop")
    basement_bin = pd.qcut(train["TotalBsmtSF"], q=5, duplicates="drop")
    built_era = pd.cut(
        train["YearBuilt"],
        bins=[1800, 1945, 1970, 1990, 2000, 2007, 2011],
        include_lowest=True,
    )
    remodel_era = pd.cut(
        train["YearRemodAdd"],
        bins=[1800, 1945, 1970, 1990, 2000, 2007, 2011],
        include_lowest=True,
    )
    age = train["YrSold"] - train["YearBuilt"]
    age_bin = pd.qcut(age, q=5, duplicates="drop")
    garage_area_bin = pd.qcut(train["GarageArea"], q=5, duplicates="drop")

    slices: list[tuple[str, pd.Series]] = [
        ("Neighborhood", train["Neighborhood"]),
        ("OverallQual", train["OverallQual"]),
        ("OverallCond", train["OverallCond"]),
        ("GrLivArea_quintile", area_bin),
        ("TotalSF_quintile", total_area_bin),
        ("TotalBsmtSF_quintile", basement_bin),
        ("YearBuilt_era", built_era),
        ("YearRemodAdd_era", remodel_era),
        ("GarageCars", train["GarageCars"]),
        ("GarageArea_quintile", garage_area_bin),
        ("KitchenQual", train["KitchenQual"]),
        ("BsmtQual", train["BsmtQual"]),
        ("ExterQual", train["ExterQual"]),
        ("SaleType", train["SaleType"]),
        ("SaleCondition", train["SaleCondition"]),
        ("YrSold", train["YrSold"]),
        ("MoSold", train["MoSold"]),
        ("MSSubClass", train["MSSubClass"]),
        (
            "Neighborhood_x_OverallQual",
            train["Neighborhood"].astype(str) + "|Q" + train["OverallQual"].astype(str),
        ),
        (
            "Neighborhood_x_GrLivArea",
            train["Neighborhood"].astype(str) + "|" + area_bin.astype(str),
        ),
        (
            "Neighborhood_x_YearBuilt",
            train["Neighborhood"].astype(str) + "|" + built_era.astype(str),
        ),
        (
            "Neighborhood_x_YearRemodAdd",
            train["Neighborhood"].astype(str) + "|" + remodel_era.astype(str),
        ),
        (
            "Neighborhood_x_SaleCondition",
            train["Neighborhood"].astype(str) + "|" + train["SaleCondition"].astype(str),
        ),
        (
            "Neighborhood_x_SaleType",
            train["Neighborhood"].astype(str) + "|" + train["SaleType"].astype(str),
        ),
        (
            "Neighborhood_x_GarageCars",
            train["Neighborhood"].astype(str) + "|G" + train["GarageCars"].fillna(-1).astype(str),
        ),
        (
            "Quality_x_GrLivArea",
            train["OverallQual"].astype(str) + "|" + area_bin.astype(str),
        ),
        (
            "Quality_x_Age",
            train["OverallQual"].astype(str) + "|" + age_bin.astype(str),
        ),
        (
            "Quality_x_RenovationEra",
            train["OverallQual"].astype(str) + "|" + remodel_era.astype(str),
        ),
        (
            "Quality_x_KitchenQual",
            train["OverallQual"].astype(str) + "|" + train["KitchenQual"].astype(str),
        ),
    ]

    rows: list[dict] = []
    for name, values in slices:
        add_slice(rows, name, values, frame)
    report = pd.DataFrame(rows).sort_values(
        ["log_rmse", "n"], ascending=[False, False]
    )
    output_path = EXPERIMENTS / "baseline_error_map.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    report.to_csv(output_path, index=False)

    errors = frame["log_error"].to_numpy()
    summary = {
        "experiment_id": "P4-BASELINE-ERROR-MAP",
        "source": "Historical baseline OOF predictions, KFold5 seed42",
        "rows": len(train),
        "pooled_log_rmse": float(np.mean(errors * errors) ** 0.5),
        "mean_log_error": float(errors.mean()),
        "dollar_rmse": float(np.mean(frame["dollar_error"] ** 2) ** 0.5),
        "mean_abs_relative_error": float(np.mean(np.abs(frame["relative_error"]))),
        "slice_count": len(report),
        "output": str(output_path.relative_to(ROOT)),
        "caveat": "Descriptive single-seed OOF map; subgroup patterns require confirmation before correction.",
    }
    summary_path = EXPERIMENTS / "baseline_error_map_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite {summary_path}")
    summary_path.write_text(pd.Series(summary).to_json(indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    if any(row.get("experiment_id") == summary["experiment_id"] for row in log_rows):
        raise ValueError("Error-map experiment already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": summary["experiment_id"],
                "hypothesis": "Error structure may cluster by location, quality, size, age, and sale context.",
                "change": "Descriptive log/dollar/relative OOF error slices and interactions",
                "cv": f"{summary['pooled_log_rmse']:.6f}",
                "cv_std": "",
                "validation_scheme": "Historical KFold5 seed42 OOF; subgroup fold-bias diagnostics",
                "runtime_seconds": "",
                "oof_correlation": "",
                "decision": "DIAGNOSTIC",
                "reason": f"Saved {summary['slice_count']} subgroup slices; descriptive only, no correction promoted",
                "status": "DIAGNOSTIC",
            }
        )
    print(pd.Series(summary).to_string())
    print("Largest-error slices:\n", report.head(12).to_string(index=False))


if __name__ == "__main__":
    main()