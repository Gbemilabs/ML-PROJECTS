from __future__ import annotations

import csv
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import phase2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SEEDS = (2027, 2028)
OUTER_SPLITS = 5
INNER_SPLITS = 4


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run fresh nested residual confirmation seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--label", default="confirm2")
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    X, _, y = phase2.prepare_raw(train, pd.read_csv(ROOT / "data" / "test.csv"))
    spec = phase2.model_specs()["XGBoost_regularized"]
    records = []
    fold_summary = []

    for seed in seeds:
        outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
        for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
            X_train, X_valid = X.iloc[outer_train], X.iloc[outer_valid]
            y_train = y.iloc[outer_train]
            inner_oof = np.zeros(len(outer_train), dtype=float)
            inner = KFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=seed * 100 + outer_fold,
            )
            for inner_train, inner_valid in inner.split(X_train):
                preprocessor = phase2.onehot_preprocessor(X_train.iloc[inner_train])
                train_matrix = preprocessor.fit_transform(X_train.iloc[inner_train])
                valid_matrix = preprocessor.transform(X_train.iloc[inner_valid])
                model = clone(spec["model"])
                model.fit(train_matrix, y_train.iloc[inner_train])
                inner_oof[inner_valid] = model.predict(valid_matrix)

            outer_preprocessor = phase2.onehot_preprocessor(X_train)
            outer_train_matrix = outer_preprocessor.fit_transform(X_train)
            outer_valid_matrix = outer_preprocessor.transform(X_valid)
            outer_model = clone(spec["model"])
            outer_model.fit(outer_train_matrix, y_train)
            base = outer_model.predict(outer_valid_matrix)

            residual_target = y_train.to_numpy() - inner_oof
            residual_preprocessor = phase2.onehot_preprocessor(X_train, scale=True)
            residual_train_matrix = residual_preprocessor.fit_transform(X_train)
            residual_valid_matrix = residual_preprocessor.transform(X_valid)
            correction_model = GradientBoostingRegressor(
                loss="huber",
                n_estimators=120,
                learning_rate=0.03,
                max_depth=1,
                min_samples_leaf=10,
                random_state=seed + outer_fold,
            )
            correction_model.fit(residual_train_matrix, residual_target)
            correction = correction_model.predict(residual_valid_matrix)
            final = base + correction
            actual = y.iloc[outer_valid].to_numpy()
            base_score = rmse(actual, base)
            final_score = rmse(actual, final)
            fold_summary.append(
                {
                    "seed": seed,
                    "fold": outer_fold,
                    "base_rmse": base_score,
                    "corrected_rmse": final_score,
                }
            )
            for offset, row_index in enumerate(outer_valid):
                records.append(
                    {
                        "Id": int(train.iloc[row_index]["Id"]),
                        "seed": seed,
                        "fold": outer_fold,
                        "y_log": float(y.iloc[row_index]),
                        "xgb_base_log": float(base[offset]),
                        "residual_correction": float(correction[offset]),
                        "corrected_log": float(final[offset]),
                    }
                )
            print(f"seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    result = pd.DataFrame(records)
    output_oof = EXPERIMENTS / f"nested_residual_{args.label}_oof.csv"
    output_report = EXPERIMENTS / f"nested_residual_{args.label}_validation.json"
    if output_oof.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite additional confirmation artifacts")
    result.to_csv(output_oof, index=False)
    actual = result["y_log"].to_numpy()
    base = result["xgb_base_log"].to_numpy()
    corrected = result["corrected_log"].to_numpy()
    fold_table = pd.DataFrame(fold_summary)
    report = {
        "experiment_id": f"P4-RESIDUAL-{args.label.upper()}",
        "hypothesis": "The fixed shallow Huber residual correction improves XGBoost across fresh outer split seeds.",
        "seeds": list(seeds),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "baseline_pooled_rmse": rmse(actual, base),
        "corrected_pooled_rmse": rmse(actual, corrected),
        "pooled_gain": rmse(actual, base) - rmse(actual, corrected),
        "baseline_mean_fold_rmse": float(fold_table.base_rmse.mean()),
        "corrected_mean_fold_rmse": float(fold_table.corrected_rmse.mean()),
        "baseline_fold_rmse_std": float(fold_table.base_rmse.std(ddof=0)),
        "corrected_fold_rmse_std": float(fold_table.corrected_rmse.std(ddof=0)),
        "baseline_worst_fold_rmse": float(fold_table.base_rmse.max()),
        "corrected_worst_fold_rmse": float(fold_table.corrected_rmse.max()),
        "per_seed": fold_table.groupby("seed").agg(
            base_mean=("base_rmse", "mean"),
            corrected_mean=("corrected_rmse", "mean"),
            base_worst=("base_rmse", "max"),
            corrected_worst=("corrected_rmse", "max"),
        ).reset_index().to_dict(orient="records"),
        "base_error_vs_correction_correlation": float(
            np.corrcoef(base - actual, result.residual_correction.to_numpy())[0, 1]
        ),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(output_oof.relative_to(ROOT)),
        "external_score": None,
        "external_champion_oof_available": False,
    }
    report["decision"] = (
        "PROMISING_LOCAL_ONLY"
        if report["pooled_gain"] >= 0.0003
        and report["corrected_worst_fold_rmse"] <= report["baseline_worst_fold_rmse"] + 0.01
        else "REJECT"
    )
    output_report.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in rows):
        raise ValueError("Confirmation experiment already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Fixed nested XGBoost + 120-tree depth-1 Huber residual correction",
                "cv": f"{report['corrected_pooled_rmse']:.6f}",
                "cv_std": f"{report['corrected_fold_rmse_std']:.6f}",
                "validation_scheme": f"Fresh nested KFold5 outer / KFold4 inner; seeds {','.join(map(str, seeds))}",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": f"{report['base_error_vs_correction_correlation']:.6f}",
                "decision": report["decision"],
                "reason": f"Pooled gain={report['pooled_gain']:.6f}; no current external champion artifact/OOF comparison",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()