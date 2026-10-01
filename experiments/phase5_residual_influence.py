from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import phase2
import phase5_shrunken_hierarchy as hierarchy

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SEEDS = (2024, 2025, 2029)
OUTER_SPLITS = 5
INNER_SPLITS = 4
FLAGGED_IDS = {524, 1299}
RIDGE_ALPHA = 10.0
MIN_FREQUENCY = 2
REPRESENTATION = "hierarchy"


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def fit_residual_ridge(
    fit_groups: pd.DataFrame,
    valid_groups: pd.DataFrame,
    residual_target: np.ndarray,
) -> np.ndarray:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=MIN_FREQUENCY,
        sparse_output=True,
    )
    train_matrix = encoder.fit_transform(fit_groups)
    valid_matrix = encoder.transform(valid_groups)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(train_matrix, residual_target)
    return model.predict(valid_matrix)


def main() -> None:
    started = time.perf_counter()
    raw_train = pd.read_csv(ROOT / "data" / "train.csv")
    raw_test = pd.read_csv(ROOT / "data" / "test.csv")
    X, _, y_series = phase2.prepare_raw(raw_train, raw_test)
    y = y_series.to_numpy(dtype=float)
    spec = phase2.model_specs()["XGBoost_regularized"]
    variants = ("all_residuals", "exclude_524_1299", "winsorize_2_98")
    records = []
    fold_records = []

    for seed in SEEDS:
        outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
        for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
            X_train, X_valid = X.iloc[outer_train], X.iloc[outer_valid]
            raw_fold_train = raw_train.iloc[outer_train]
            raw_fold_valid = raw_train.iloc[outer_valid]
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
            fit_groups, valid_groups = hierarchy.group_frame(
                raw_fold_train,
                raw_fold_valid,
                REPRESENTATION,
            )
            correction_values = {}

            correction_values["all_residuals"] = fit_residual_ridge(
                fit_groups,
                valid_groups,
                residual,
            )

            keep = ~raw_fold_train["Id"].isin(FLAGGED_IDS).to_numpy()
            correction_values["exclude_524_1299"] = fit_residual_ridge(
                fit_groups.loc[keep].reset_index(drop=True),
                valid_groups,
                residual[keep],
            )

            lower, upper = np.quantile(residual, [0.02, 0.98])
            clipped = np.clip(residual, lower, upper)
            correction_values["winsorize_2_98"] = fit_residual_ridge(
                fit_groups,
                valid_groups,
                clipped,
            )

            actual = y[outer_valid]
            fold_record = {
                "seed": seed,
                "fold": outer_fold,
                "base_rmse": rmse(actual, base),
            }
            for variant, correction in correction_values.items():
                final = base + correction
                fold_record[f"{variant}_rmse"] = rmse(actual, final)
                for offset, row_index in enumerate(outer_valid):
                    records.append(
                        {
                            "Id": int(raw_train.iloc[row_index]["Id"]),
                            "seed": seed,
                            "fold": outer_fold,
                            "y_log": float(y[row_index]),
                            "xgb_base_log": float(base[offset]),
                            "variant": variant,
                            "residual_correction": float(correction[offset]),
                            "final_log": float(final[offset]),
                        }
                    )
            fold_records.append(fold_record)
            print(f"seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    oof = pd.DataFrame(records)
    output_oof = EXPERIMENTS / "phase5_residual_influence_oof.csv"
    output_report = EXPERIMENTS / "phase5_residual_influence_validation.json"
    if output_oof.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite phase-five residual influence artifacts")
    oof.to_csv(output_oof, index=False)

    summary = {}
    per_seed = {}
    for variant in variants:
        data = oof.loc[oof.variant.eq(variant)]
        errors = data.final_log.to_numpy() - data.y_log.to_numpy()
        baseline_errors = data.xgb_base_log.to_numpy() - data.y_log.to_numpy()
        folds = pd.DataFrame(fold_records)
        fold_scores = folds[f"{variant}_rmse"].to_numpy()
        summary[variant] = {
            "baseline_pooled_rmse": float(np.mean(baseline_errors**2) ** 0.5),
            "corrected_pooled_rmse": float(np.mean(errors**2) ** 0.5),
            "pooled_gain": float(np.mean(baseline_errors**2) ** 0.5 - np.mean(errors**2) ** 0.5),
            "mean_fold_rmse": float(fold_scores.mean()),
            "fold_rmse_std": float(fold_scores.std()),
            "worst_fold_rmse": float(fold_scores.max()),
        }
        for seed, group in data.groupby("seed"):
            per_seed.setdefault(variant, {})[str(int(seed))] = {
                "baseline_rmse": rmse(group.y_log.to_numpy(), group.xgb_base_log.to_numpy()),
                "corrected_rmse": rmse(group.y_log.to_numpy(), group.final_log.to_numpy()),
            }

    candidate_rows = oof.loc[oof.variant.eq("all_residuals")].copy()
    flagged = candidate_rows.Id.isin(FLAGGED_IDS).to_numpy()
    actual_flag = candidate_rows.y_log.to_numpy()
    base_flag = candidate_rows.xgb_base_log.to_numpy()
    corrected_flag = candidate_rows.final_log.to_numpy()
    influence = {
        "all_rows": {
            "baseline_rmse": rmse(actual_flag, base_flag),
            "corrected_rmse": rmse(actual_flag, corrected_flag),
        },
        "excluding_524_1299_from_validation": {
            "baseline_rmse": rmse(actual_flag[~flagged], base_flag[~flagged]),
            "corrected_rmse": rmse(actual_flag[~flagged], corrected_flag[~flagged]),
        },
    }
    report = {
        "experiment_id": "P5-RESIDUAL-INFLUENCE-ABLATION",
        "hypothesis": "The champion-transfer hierarchy may depend on the two extreme Edwards residuals; winsorization or excluding them from residual fitting should reveal that dependence.",
        "base_model": "Regularized XGBoost with nested inner-OOF residual labels",
        "residual_hierarchy": {"representation": REPRESENTATION, "ridge_alpha": RIDGE_ALPHA, "min_frequency": MIN_FREQUENCY},
        "variants": list(variants),
        "seeds": list(SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "metrics": summary,
        "per_seed": per_seed,
        "validation_influence": influence,
        "oof_path": str(output_oof.relative_to(ROOT)),
        "decision": "REVIEW; do not promote without champion-transfer stability",
        "external_score": None,
    }
    output_report.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Frozen hierarchical residual; exclude two Edwards rows or winsorize residual target",
                "cv": f"{summary['exclude_524_1299']['corrected_pooled_rmse']:.6f}",
                "cv_std": f"{summary['exclude_524_1299']['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested KFold5 outer / KFold4 inner; seeds 2024,2025,2029",
                "runtime_seconds": round(time.perf_counter() - started, 2),
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"All-data gain={summary['all_residuals']['pooled_gain']:.6f}; exclude-flagged correction gain={summary['exclude_524_1299']['pooled_gain']:.6f}; winsorized gain={summary['winsorize_2_98']['pooled_gain']:.6f}",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()