from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import phase2
import phase4_hierarchical_residual as hierarchy

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SEEDS = (2029, 2030)
OUTER_SPLITS = 5
INNER_SPLITS = 4
GROUP_NAME = "Neighborhood_x_OverallQual"
SHRINKAGE = 5.0


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def summary(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    errors = frame[prediction_column].to_numpy() - frame["y_log"].to_numpy()
    fold_scores = frame.assign(error=errors).groupby(["seed", "fold"]).error.apply(
        lambda values: float(np.mean(values.to_numpy() ** 2) ** 0.5)
    )
    return {
        "pooled_rmse": float(np.mean(errors * errors) ** 0.5),
        "mean_fold_rmse": float(fold_scores.mean()),
        "fold_rmse_std": float(fold_scores.std(ddof=0)),
        "worst_fold_rmse": float(fold_scores.max()),
        "median_fold_rmse": float(fold_scores.median()),
    }


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    X, _, y = phase2.prepare_raw(train, pd.read_csv(ROOT / "data" / "test.csv"))
    y = y.to_numpy(dtype=float)
    spec = phase2.model_specs()["XGBoost_regularized"]
    records = []

    for seed in SEEDS:
        outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
        for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
            X_train, X_valid = X.iloc[outer_train], X.iloc[outer_valid]
            raw_train, raw_valid = train.iloc[outer_train], train.iloc[outer_valid]
            y_train = y[outer_train]
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
                model.fit(train_matrix, y_train[inner_train])
                inner_oof[inner_valid] = model.predict(valid_matrix)

            outer_preprocessor = phase2.onehot_preprocessor(X_train)
            outer_train_matrix = outer_preprocessor.fit_transform(X_train)
            outer_valid_matrix = outer_preprocessor.transform(X_valid)
            outer_model = clone(spec["model"])
            outer_model.fit(outer_train_matrix, y_train)
            base = outer_model.predict(outer_valid_matrix)
            residual = y_train - inner_oof
            fit_keys, valid_keys = hierarchy.group_keys(raw_train, raw_valid)
            correction, count = hierarchy.group_residual_correction(
                fit_keys[GROUP_NAME],
                valid_keys[GROUP_NAME],
                residual,
                SHRINKAGE,
            )
            corrected = base + correction
            actual = y[outer_valid]
            for offset, row_index in enumerate(outer_valid):
                records.append(
                    {
                        "Id": int(train.iloc[row_index]["Id"]),
                        "seed": seed,
                        "fold": outer_fold,
                        "y_log": float(y[row_index]),
                        "xgb_base_log": float(base[offset]),
                        "residual_correction": float(correction[offset]),
                        "group_comparable_count": float(count[offset]),
                        "corrected_log": float(corrected[offset]),
                    }
                )
            print(f"seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    result = pd.DataFrame(records)
    output_oof = EXPERIMENTS / "hierarchical_residual_confirm_oof.csv"
    output_report = EXPERIMENTS / "hierarchical_residual_confirm_validation.json"
    if output_oof.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite hierarchical-residual confirmation artifacts")
    result.to_csv(output_oof, index=False)

    original = pd.read_csv(EXPERIMENTS / "hierarchical_residual_oof.csv")
    original_confirmation = original.loc[original.stage.eq("confirmation")].rename(
        columns={"selected_group_residual_log": "corrected_log"}
    )
    combined = pd.concat(
        [original_confirmation, result],
        ignore_index=True,
        sort=False,
    )
    base_metrics = summary(combined, "xgb_base_log")
    corrected_metrics = summary(combined, "corrected_log")
    gain = base_metrics["pooled_rmse"] - corrected_metrics["pooled_rmse"]
    worst_change = corrected_metrics["worst_fold_rmse"] - base_metrics["worst_fold_rmse"]
    report = {
        "experiment_id": "P4-HIERARCHICAL-RESIDUAL-CONFIRM2",
        "frozen_config_from_discovery": {"group": GROUP_NAME, "shrinkage": SHRINKAGE},
        "new_confirmation_seeds": list(SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "new_seed_base": summary(result, "xgb_base_log"),
        "new_seed_corrected": summary(result, "corrected_log"),
        "per_seed": {
            str(int(seed)): {
                "base": summary(rows, "xgb_base_log"),
                "corrected": summary(rows, "corrected_log"),
            }
            for seed, rows in result.groupby("seed")
        },
        "combined_confirmation_seeds": [2025, 2026, 2027, 2029, 2030],
        "combined_base": base_metrics,
        "combined_corrected": corrected_metrics,
        "combined_pooled_gain": gain,
        "combined_worst_fold_change": worst_change,
        "local_shadow_gate": "pooled RMSE gain >= 0.0003 and worst fold <= base worst + 0.01",
        "local_shadow_gate_passed": bool(gain >= 0.0003 and worst_change <= 0.01),
        "existing_shadow_candidate": "experiments/shadow_candidates/hierarchical_residual_xgb.csv",
        "external_champion_score": 0.12654,
        "external_score": None,
        "decision": "PROMISING_LOCAL_XGB_ONLY" if gain >= 0.0003 and worst_change <= 0.01 else "REJECT",
        "caveat": "The experiment is XGBoost-only and has not been evaluated as a correction to the frozen CatBoost/XGBoost champion.",
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(output_oof.relative_to(ROOT)),
    }
    output_report.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in rows):
        raise ValueError("Hierarchical confirmation already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": "The frozen neighborhood-quality residual effect transfers to two additional outer split seeds.",
                "change": "No retuning; neighborhood x quality residual mean, shrinkage 5",
                "cv": f"{corrected_metrics['pooled_rmse']:.6f}",
                "cv_std": f"{corrected_metrics['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested KFold5 outer / KFold4 inner; confirmation seeds 2029,2030 plus prior confirmation",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Combined five-seed confirmation gain={gain:.6f}; worst-fold change={worst_change:+.6f}; XGBoost-only, no champion-ensemble test",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()