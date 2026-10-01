from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import phase2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (42, 117)
CONFIRMATION_SEEDS = (2026,)
OUTER_SPLITS = 5
INNER_SPLITS = 4


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def summarize(rows: pd.DataFrame, stage: str, prediction_column: str) -> dict[str, float]:
    selected = rows.loc[rows["stage"].eq(stage)]
    errors = selected[prediction_column].to_numpy() - selected["y_log"].to_numpy()
    fold_scores = selected.groupby(["seed", "fold"], sort=False).apply(
        lambda fold: rmse(fold["y_log"].to_numpy(), fold[prediction_column].to_numpy()),
        include_groups=False,
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
    xgb_spec = phase2.model_specs()["XGBoost_regularized"]
    output_rows: list[dict] = []

    for stage, seeds in (("discovery", DISCOVERY_SEEDS), ("confirmation", CONFIRMATION_SEEDS)):
        for seed in seeds:
            outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
            for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
                X_outer_train = X.iloc[outer_train]
                X_outer_valid = X.iloc[outer_valid]
                y_outer_train = y.iloc[outer_train]
                inner_oof = np.zeros(len(outer_train), dtype=float)
                inner = KFold(
                    n_splits=INNER_SPLITS,
                    shuffle=True,
                    random_state=seed * 100 + outer_fold,
                )

                for inner_train, inner_valid in inner.split(X_outer_train):
                    X_inner_train = X_outer_train.iloc[inner_train]
                    X_inner_valid = X_outer_train.iloc[inner_valid]
                    preprocessor = phase2.onehot_preprocessor(X_inner_train)
                    train_matrix = preprocessor.fit_transform(X_inner_train)
                    valid_matrix = preprocessor.transform(X_inner_valid)
                    model = clone(xgb_spec["model"])
                    model.fit(train_matrix, y_outer_train.iloc[inner_train])
                    inner_oof[inner_valid] = model.predict(valid_matrix)

                outer_preprocessor = phase2.onehot_preprocessor(X_outer_train)
                outer_train_matrix = outer_preprocessor.fit_transform(X_outer_train)
                outer_valid_matrix = outer_preprocessor.transform(X_outer_valid)
                outer_model = clone(xgb_spec["model"])
                outer_model.fit(outer_train_matrix, y_outer_train)
                base_prediction = outer_model.predict(outer_valid_matrix)

                # Inner OOF residuals are generated entirely within the outer training set.
                residual_target = y_outer_train.to_numpy() - inner_oof
                residual_preprocessor = phase2.onehot_preprocessor(X_outer_train, scale=True)
                residual_train_matrix = residual_preprocessor.fit_transform(X_outer_train)
                residual_valid_matrix = residual_preprocessor.transform(X_outer_valid)
                residual_models = {
                    "ridge10": Ridge(alpha=10.0),
                    "ridge50": Ridge(alpha=50.0),
                    "gradient_boosting": GradientBoostingRegressor(
                        loss="huber",
                        n_estimators=120,
                        learning_rate=0.03,
                        max_depth=1,
                        min_samples_leaf=10,
                        random_state=seed + outer_fold,
                    ),
                }
                corrections = {}
                for name, residual_model in residual_models.items():
                    residual_model.fit(residual_train_matrix, residual_target)
                    corrections[name] = residual_model.predict(residual_valid_matrix)

                for row_offset, row_index in enumerate(outer_valid):
                    record = {
                        "Id": int(train.iloc[row_index]["Id"]),
                        "stage": stage,
                        "seed": seed,
                        "fold": outer_fold,
                        "y_log": float(y.iloc[row_index]),
                        "xgb_base_log": float(base_prediction[row_offset]),
                    }
                    for name, correction in corrections.items():
                        record[f"{name}_correction"] = float(correction[row_offset])
                        record[f"{name}_final_log"] = float(
                            base_prediction[row_offset] + correction[row_offset]
                        )
                    output_rows.append(record)
                print(
                    f"{stage} seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}",
                    flush=True,
                )

    results = pd.DataFrame(output_rows)
    oof_path = EXPERIMENTS / "nested_residual_oof.csv"
    report_path = EXPERIMENTS / "nested_residual_validation.json"
    if oof_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite prior nested-residual artifacts")
    results.to_csv(oof_path, index=False)

    prediction_columns = {
        "xgb_base": "xgb_base_log",
        "ridge10": "ridge10_final_log",
        "ridge50": "ridge50_final_log",
        "gradient_boosting": "gradient_boosting_final_log",
    }
    metrics = {
        stage: {
            name: summarize(results, stage, column)
            for name, column in prediction_columns.items()
        }
        for stage in ("discovery", "confirmation")
    }
    confirmation = results.loc[results["stage"].eq("confirmation")]
    error_correlations = confirmation[
        ["xgb_base_log", "ridge10_final_log", "ridge50_final_log", "gradient_boosting_final_log"]
    ].sub(confirmation["y_log"], axis=0).corr().to_dict()
    correction_correlations = {
        name: float(
            np.corrcoef(
                confirmation["xgb_base_log"] - confirmation["y_log"],
                confirmation[correction],
            )[0, 1]
        )
        for name, correction in (
            ("ridge10", "ridge10_correction"),
            ("ridge50", "ridge50_correction"),
            ("gradient_boosting", "gradient_boosting_correction"),
        )
    }
    report = {
        "experiment_id": "P4-NESTED-RESIDUAL",
        "hypothesis": "Inner-fold residuals may contain feature-predictable error that transfers to an untouched outer fold.",
        "model": "XGBoost_regularized baseline with Ridge and shallow Huber GradientBoosting residual learners",
        "residual_target_firewall": "For every outer validation fold, base residual labels come only from inner OOF predictions trained within the outer training rows.",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "metrics": metrics,
        "confirmation_residual_error_correlations": error_correlations,
        "confirmation_base_error_vs_correction_correlations": correction_correlations,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "external_champion_score": 0.12654,
        "external_champion_oof_available": False,
        "external_score": None,
        "decision": "REVIEW_CONFIRMATION_METRICS; no external champion comparison possible",
    }
    report_path.write_text(json.dumps(report, indent=2))

    best_name = min(
        ("ridge10", "ridge50", "gradient_boosting"),
        key=lambda name: metrics["confirmation"][name]["pooled_rmse"],
    )
    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in log_rows):
        raise ValueError("Nested-residual experiment already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Outer-fold XGBoost plus inner-OOF residual Ridge/Huber learners",
                "cv": f"{metrics['confirmation'][best_name]['pooled_rmse']:.6f}",
                "cv_std": f"{metrics['confirmation'][best_name]['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested outer KFold5; inner KFold4; discovery 2 seeds, confirmation 1 seed",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": "INVESTIGATE",
                "reason": f"Best confirmation correction={best_name}; compared with nested XGBoost only, not the current external champion",
                "status": "INVESTIGATE",
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()