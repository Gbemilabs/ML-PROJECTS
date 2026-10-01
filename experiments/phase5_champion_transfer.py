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
import phase5_shrunken_hierarchy as hierarchy

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (19, 43, 89)
CONFIRMATION_SEEDS = (2024, 2025, 2029)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTED_GROUP = "hierarchy"
RIDGE_ALPHA = 10.0
MIN_FREQUENCY = 2
NESTED_SPLITS = 5


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def summarize(frame: pd.DataFrame, prediction: str) -> dict[str, float]:
    error = frame[prediction].to_numpy() - frame["y_log"].to_numpy()
    fold_scores = frame.assign(error=error).groupby(["seed", "fold"]).error.apply(
        lambda values: float(np.mean(values.to_numpy() ** 2) ** 0.5)
    )
    return {
        "pooled_rmse": float(np.mean(error * error) ** 0.5),
        "mean_fold_rmse": float(fold_scores.mean()),
        "fold_rmse_std": float(fold_scores.std(ddof=0)),
        "worst_fold_rmse": float(fold_scores.max()),
        "median_fold_rmse": float(fold_scores.median()),
    }


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_seed42.csv")
    if not np.array_equal(champion_oof["Id"].to_numpy(), train["Id"].to_numpy()):
        raise ValueError("Champion OOF IDs must align exactly with training rows")
    champion_log = np.log1p(champion_oof["SalePrice"].to_numpy(dtype=float))

    source_oof = pd.read_csv(EXPERIMENTS / "phase5_hierarchical_residual_oof.csv")
    correction_rows = source_oof.loc[source_oof.stage.eq("discovery") | source_oof.stage.eq("confirmation")].copy()
    correction_rows["residual_offset"] = (
        correction_rows["stage_prediction"] - correction_rows["xgb_base_log"]
    )
    champion_by_id = dict(zip(train["Id"], champion_log))
    actual_by_id = dict(zip(train["Id"], np.log1p(train["SalePrice"].to_numpy(dtype=float))))
    correction_rows["champion_oof_log"] = correction_rows["Id"].map(champion_by_id)
    correction_rows["y_log"] = correction_rows["Id"].map(actual_by_id)

    discovery = correction_rows.loc[correction_rows.stage.eq("discovery")].copy()
    confirmation = correction_rows.loc[correction_rows.stage.eq("confirmation")].copy()
    discovery_metrics = {
        alpha: summarize(
            discovery.assign(candidate=discovery.champion_oof_log + alpha * discovery.residual_offset),
            "candidate",
        )
        for alpha in ALPHAS
    }
    selected_alpha = min(ALPHAS, key=lambda alpha: discovery_metrics[alpha]["pooled_rmse"])
    confirmation_metrics = {
        alpha: summarize(
            confirmation.assign(candidate=confirmation.champion_oof_log + alpha * confirmation.residual_offset),
            "candidate",
        )
        for alpha in ALPHAS
    }
    confirmation_base = summarize(confirmation, "champion_oof_log")
    selected_confirm = confirmation_metrics[selected_alpha]
    gain = confirmation_base["pooled_rmse"] - selected_confirm["pooled_rmse"]
    worst_change = selected_confirm["worst_fold_rmse"] - confirmation_base["worst_fold_rmse"]
    by_seed = {}
    for seed, rows in confirmation.groupby("seed"):
        candidate_prediction = rows["champion_oof_log"].to_numpy() + selected_alpha * rows[
            "residual_offset"
        ].to_numpy()
        by_seed[str(int(seed))] = {
            "baseline_rmse": rmse(rows["y_log"].to_numpy(), rows["champion_oof_log"].to_numpy()),
            "candidate_rmse": rmse(rows["y_log"].to_numpy(), candidate_prediction),
        }
    positive_seed_count = sum(value["candidate_rmse"] < value["baseline_rmse"] for value in by_seed.values())
    eligible = gain >= 0.0003 and worst_change <= 0.01 and positive_seed_count >= 2

    output_oof = EXPERIMENTS / "phase5_champion_transfer_oof.csv"
    output_report = EXPERIMENTS / "phase5_champion_transfer_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase5_champion_residual_transfer.csv"
    components_path = EXPERIMENTS / "phase5_champion_transfer_test_components.csv"
    if output_oof.exists() or output_report.exists() or (
        eligible and (candidate_path.exists() or components_path.exists())
    ):
        raise FileExistsError("Refusing to overwrite phase-five champion-transfer artifacts")
    correction_rows["transfer_candidate_log"] = (
        correction_rows["champion_oof_log"] + selected_alpha * correction_rows["residual_offset"]
    )
    correction_rows.to_csv(output_oof, index=False)

    candidate_written = False
    if eligible:
        X, X_test, y = phase2.prepare_raw(train, test)
        spec = phase2.model_specs()["XGBoost_regularized"]
        full_preprocessor = phase2.onehot_preprocessor(X)
        train_matrix = full_preprocessor.fit_transform(X)
        test_matrix = full_preprocessor.transform(X_test)
        full_model = clone(spec["model"])
        full_model.fit(train_matrix, y)
        xgb_test = full_model.predict(test_matrix)

        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=4, shuffle=True, random_state=2031)
        for inner_train, inner_valid in inner.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            inner_train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            inner_valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(spec["model"])
            model.fit(inner_train_matrix, y.iloc[inner_train])
            inner_oof[inner_valid] = model.predict(inner_valid_matrix)
        fit_groups, test_groups = hierarchy.group_frame(train, test, SELECTED_GROUP)
        correction_model = hierarchy.make_correction(
            fit_groups,
            test_groups,
            y.to_numpy() - inner_oof,
            RIDGE_ALPHA,
            MIN_FREQUENCY,
        )
        champion_submission = pd.read_csv(
            EXPERIMENTS / "champion" / "submission_0.12654.csv"
        )
        champion_test_log = np.log1p(champion_submission["SalePrice"].to_numpy(dtype=float))
        candidate_log = champion_test_log + selected_alpha * correction_model
        candidate_price = np.expm1(candidate_log)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Champion-transfer candidate has invalid test prices")
        shadow_dir = candidate_path.parent
        shadow_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test["Id"], "SalePrice": candidate_price}).to_csv(
            candidate_path,
            index=False,
        )
        pd.DataFrame(
            {
                "Id": test["Id"],
                "champion_log": champion_test_log,
                "xgb_log": xgb_test,
                "residual_offset_log": correction_model,
                "scale": selected_alpha,
                "candidate_log": candidate_log,
                "champion_price": champion_submission["SalePrice"],
                "candidate_price": candidate_price,
            }
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P5-CHAMPION-RESIDUAL-TRANSFER",
        "hypothesis": "A leakage-safe XGBoost residual hierarchy transfers to the protected champion's OOF residuals.",
        "base_oof": "experiments/champion/oof_seed42.csv; saved champion OOF predictions are cross-fitted at seed 42",
        "correction_oof": "Inner-OOF residual hierarchy predictions from phase5_hierarchical_residual_oof.csv; each correction model excludes its outer validation targets",
        "group_representation": SELECTED_GROUP,
        "ridge_alpha": RIDGE_ALPHA,
        "minimum_frequency": MIN_FREQUENCY,
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selected_scale_on_discovery_only": selected_alpha,
        "discovery_by_scale": {str(alpha): values for alpha, values in discovery_metrics.items()},
        "confirmation_baseline": confirmation_base,
        "confirmation_candidate": selected_confirm,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": by_seed,
        "confirmation_seeds_improved": positive_seed_count,
        "shadow_gate": "Pooled champion-OOF gain >=0.0003, worst fold <=baseline+0.01, and at least two of three seeds improve",
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "components_path": str(components_path.relative_to(ROOT)) if candidate_written else None,
        "external_champion_score": 0.12654,
        "external_score": None,
        "decision": "SHADOW_ONLY_CHAMPION_TRANSFER" if eligible else "REJECT",
        "caveat": "The correction is estimated from regularized-XGBoost residuals and transferred to the champion; confirmation uses the champion's fixed seed-42 OOF predictions and is not an independent repeated OOF of the full CatBoost/XGBoost pipeline.",
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(output_oof.relative_to(ROOT)),
    }
    output_report.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in rows):
        raise ValueError("Champion-transfer experiment already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Apply nested XGBoost hierarchy correction to protected champion OOF; scale chosen on discovery seeds",
                "cv": f"{selected_confirm['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirm['fold_rmse_std']:.6f}",
                "validation_scheme": "Champion OOF transfer; discovery/confirmation correction seeds; nested base residual firewall",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected residual scale={selected_alpha:g}; confirmation gain={gain:.6f}; caveat: fixed champion OOF seed42, not repeated champion OOF",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()