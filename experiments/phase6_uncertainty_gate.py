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
DISCOVERY_SEEDS = (2031, 2032)
CONFIRMATION_SEEDS = (2033, 2034, 2035)
NESTED_SPLITS = 5
INNER_SPLITS = 4
UNCERTAINTY_QUANTILES = (0.75, 0.85, 0.9, 0.95)
INCREMENTAL_SCALES = (0.25, 0.5, 1.0)
RIDGE_ALPHA = 10.0
MIN_FREQUENCY = 10


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


def fit_correction(
    train_groups: pd.DataFrame,
    query_groups: pd.DataFrame,
    residual: np.ndarray,
) -> np.ndarray:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=MIN_FREQUENCY,
        sparse_output=True,
    )
    train_matrix = encoder.fit_transform(train_groups)
    query_matrix = encoder.transform(query_groups)
    model = Ridge(alpha=RIDGE_ALPHA).fit(train_matrix, residual)
    return model.predict(query_matrix)


def summarize(
    actual: np.ndarray,
    prediction: np.ndarray,
    fold_assignment: np.ndarray,
) -> dict[str, float]:
    error = prediction - actual
    fold_rmse = [
        rmse(actual[fold_assignment == fold], prediction[fold_assignment == fold])
        for fold in np.unique(fold_assignment)
    ]
    return {
        "pooled_rmse": float(np.mean(error * error) ** 0.5),
        "mean_fold_rmse": float(np.mean(fold_rmse)),
        "fold_rmse_std": float(np.std(fold_rmse)),
        "worst_fold_rmse": float(np.max(fold_rmse)),
        "median_fold_rmse": float(np.median(fold_rmse)),
    }


def stage_correction_table(
    source_oof: pd.DataFrame,
    stage: str,
) -> pd.DataFrame:
    stage_rows = source_oof.loc[source_oof.stage.eq(stage)]
    return (
        stage_rows.groupby("Id").correction_a10_m10.agg(["mean", "std", "count"])
        .rename(columns={"mean": "correction_mean", "std": "correction_sd", "count": "seed_count"})
    )


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
    if champion_oof.Id.duplicated().any() or set(champion_oof.Id) != set(train.Id):
        raise ValueError("Champion OOF must contain each training ID exactly once")
    champion_oof = champion_oof.set_index("Id").reindex(train.Id).reset_index()
    if not np.array_equal(champion_submission.Id.to_numpy(), test.Id.to_numpy()):
        raise ValueError("Champion submission IDs must align with test data")
    if not np.array_equal(
        pd.read_csv(ROOT / "submission.csv").Id.to_numpy(),
        champion_submission.Id.to_numpy(),
    ):
        raise ValueError("Root submission is not aligned with the protected 0.12374 champion")

    source_oof = pd.read_csv(EXPERIMENTS / "phase5_support_floor_transfer_oof.csv")
    discovery = stage_correction_table(source_oof, "discovery").reindex(train.Id)
    confirmation = stage_correction_table(source_oof, "confirmation").reindex(train.Id)
    y = champion_oof.y_log.to_numpy(dtype=float)
    base = champion_oof.candidate_oof_log.to_numpy(dtype=float)
    fold_ids = np.empty(len(train), dtype=int)
    for fold, (_, valid_idx) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=42).split(train)
    ):
        fold_ids[valid_idx] = fold

    thresholds = {
        quantile: float(discovery.correction_sd.fillna(0).quantile(quantile))
        for quantile in UNCERTAINTY_QUANTILES
    }
    configs = [(quantile, scale) for quantile in UNCERTAINTY_QUANTILES for scale in INCREMENTAL_SCALES]
    discovery_scores = {}
    confirmation_scores = {}
    for quantile, increment in configs:
        threshold = thresholds[quantile]
        discovery_gate = discovery.correction_sd.fillna(0).to_numpy() >= threshold
        confirmation_gate = confirmation.correction_sd.fillna(0).to_numpy() >= threshold
        discovery_prediction = base + increment * discovery.correction_mean.to_numpy() * discovery_gate
        confirmation_prediction = base + increment * confirmation.correction_mean.to_numpy() * confirmation_gate
        discovery_scores[(quantile, increment)] = summarize(y, discovery_prediction, fold_ids)
        confirmation_scores[(quantile, increment)] = summarize(y, confirmation_prediction, fold_ids)

    selected = min(configs, key=lambda config: discovery_scores[config]["pooled_rmse"])
    quantile, increment = selected
    threshold = thresholds[quantile]
    confirmation_gate = confirmation.correction_sd.fillna(0).to_numpy() >= threshold
    confirmation_prediction = base + increment * confirmation.correction_mean.to_numpy() * confirmation_gate
    confirmation_metric = confirmation_scores[selected]
    base_metric = summarize(y, base, fold_ids)
    gain = base_metric["pooled_rmse"] - confirmation_metric["pooled_rmse"]
    worst_change = confirmation_metric["worst_fold_rmse"] - base_metric["worst_fold_rmse"]
    gate_passed = gain >= 0.0003 and worst_change <= 0.01
    confirmation_seed_results = {}
    confirmation_source = source_oof.loc[source_oof.stage.eq("confirmation")]
    for seed, seed_rows in confirmation_source.groupby("seed"):
        seed_correction = seed_rows.set_index("Id").correction_a10_m10.reindex(train.Id).to_numpy()
        seed_prediction = base + increment * seed_correction * confirmation_gate
        confirmation_seed_results[str(int(seed))] = {
            "champion_oof_rmse": rmse(y, base),
            "candidate_rmse": rmse(y, seed_prediction),
        }

    output_oof = EXPERIMENTS / "phase6_uncertainty_gate_oof.csv"
    output_report = EXPERIMENTS / "phase6_uncertainty_gate_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase6_uncertainty_gate.csv"
    components_path = EXPERIMENTS / "phase6_uncertainty_gate_test_components.csv"
    if output_oof.exists() or output_report.exists() or candidate_path.exists() or components_path.exists():
        raise FileExistsError("Refusing to overwrite phase-six uncertainty artifacts")

    oof_frame = pd.DataFrame(
        {
            "Id": train.Id,
            "y_log": y,
            "current_champion_oof_log": base,
            "discovery_correction_mean": discovery.correction_mean.to_numpy(),
            "discovery_correction_sd": discovery.correction_sd.to_numpy(),
            "confirmation_correction_mean": confirmation.correction_mean.to_numpy(),
            "confirmation_correction_sd": confirmation.correction_sd.to_numpy(),
            "confirmation_gate": confirmation_gate,
            "uncertainty_candidate_oof_log": confirmation_prediction,
        }
    )
    oof_frame.to_csv(output_oof, index=False)

    X, X_test, y_series = phase2.prepare_raw(train, test)
    y_model = y_series.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]
    test_corrections = []
    for seed in (2031, 2032, 2033, 2034, 2035):
        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed * 100)
        for inner_train, inner_valid in inner.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"]).fit(train_matrix, y_model[inner_train])
            inner_oof[inner_valid] = model.predict(valid_matrix)
        fit_groups, test_groups = hierarchy.group_frame(train, test, "hierarchy")
        test_corrections.append(
            fit_correction(fit_groups, test_groups, y_model - inner_oof)
        )
    test_corrections = np.asarray(test_corrections)
    test_correction_mean = test_corrections.mean(axis=0)
    test_correction_sd = test_corrections.std(axis=0, ddof=0)
    test_gate = test_correction_sd >= threshold
    external_base = np.log1p(champion_submission.SalePrice.to_numpy(dtype=float))
    candidate_log = external_base + increment * test_correction_mean * test_gate
    candidate_price = np.expm1(candidate_log)
    if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
        raise ValueError("Uncertainty-gated candidate contains invalid prices")
    pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
    pd.DataFrame(
        {
            "Id": test.Id,
            "champion_price": champion_submission.SalePrice,
            "correction_mean": test_correction_mean,
            "correction_sd": test_correction_sd,
            "gate_threshold": threshold,
            "gate_active": test_gate,
            "incremental_scale": increment,
            "candidate_price": candidate_price,
        }
    ).to_csv(components_path, index=False)

    report = {
        "experiment_id": "P6-UNCERTAINTY-GATED-RESIDUAL",
        "hypothesis": "Cross-seed uncertainty of the local residual effect can gate a small extra correction in high-confidence regions.",
        "parent_champion": "0.12374 external system; preserved under experiments/champion/",
        "correction_model": "Ridge alpha10, hierarchy one-hot min_frequency10, trained on inner-OOF XGBoost residuals",
        "discovery_seeds": [2031, 2032],
        "confirmation_seeds": [2033, 2034, 2035],
        "threshold_quantiles": list(UNCERTAINTY_QUANTILES),
        "incremental_scales": list(INCREMENTAL_SCALES),
        "discovery_selected_only": {"uncertainty_quantile": quantile, "threshold": threshold, "increment": increment},
        "discovery_metrics": discovery_scores[selected],
        "confirmation_baseline": base_metric,
        "confirmation_candidate": confirmation_metric,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": confirmation_seed_results,
        "predeclared_gate_passed": bool(gate_passed),
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "components_path": str(components_path.relative_to(ROOT)),
        "prediction_movement": {
            "mean_abs_log_delta": float(np.mean(np.abs(candidate_log - external_base))),
            "p95_abs_log_delta": float(np.quantile(np.abs(candidate_log - external_base), 0.95)),
            "max_abs_log_delta": float(np.max(np.abs(candidate_log - external_base))),
            "test_rows_gated": int(test_gate.sum()),
            "test_rows_total": len(test),
        },
        "external_score": None,
        "decision": "SHADOW_ONLY" if gate_passed else "REJECT",
        "caveat": "Discovery/confirmation seeds reuse the same homes; this is seed-robustness evidence, not an independent sample or Kaggle result.",
        "oof_path": str(output_oof.relative_to(ROOT)),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    output_report.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Uncertainty-gated incremental residual correction from cross-seed local-effect dispersion",
                "cv": f"{confirmation_metric['pooled_rmse']:.6f}",
                "cv_std": f"{confirmation_metric['fold_rmse_std']:.6f}",
                "validation_scheme": "Discovery correction seeds 2031/2032; confirmation 2033/2034/2035; fixed champion OOF",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Discovery chose q={quantile}, increment={increment}; confirmation gain={gain:.6f}; external score pending",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()