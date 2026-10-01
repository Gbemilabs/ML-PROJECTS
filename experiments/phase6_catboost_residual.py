from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import phase2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (23, 47, 83)
CONFIRMATION_SEEDS = (2036, 2037, 2041)
OUTER_SPLITS = 5
INNER_SPLITS = 4
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 600,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 10.0,
    "random_strength": 0.5,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 4,
}


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


def catboost_frame(frame: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in categorical_columns:
        result[column] = result[column].fillna("Missing").astype(str)
    return result


def summarize(actual: list[float], prediction: list[float], fold_scores: list[float]) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(actual)
    return {
        "pooled_rmse": float(np.mean(error * error) ** 0.5),
        "mean_fold_rmse": float(np.mean(fold_scores)),
        "fold_rmse_std": float(np.std(fold_scores)),
        "worst_fold_rmse": float(np.max(fold_scores)),
        "median_fold_rmse": float(np.median(fold_scores)),
    }


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
    if champion_oof.Id.duplicated().any() or set(champion_oof.Id) != set(train.Id):
        raise ValueError("Champion OOF must contain every training ID exactly once")
    champion_oof = champion_oof.set_index("Id").reindex(train.Id).reset_index()
    if not np.array_equal(champion_submission.Id, test.Id):
        raise ValueError("Champion submission IDs do not match test order")

    X, X_test, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    champion_log = champion_oof.candidate_oof_log.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]
    categorical_columns = [column for column in X.columns if not pd.api.types.is_numeric_dtype(X[column])]
    scales = list(SCALES)
    actual_by_stage = {"discovery": [], "confirmation": []}
    base_by_stage = {"discovery": [], "confirmation": []}
    corrections = {"discovery": [], "confirmation": []}
    fold_scores = {"discovery": {scale: [] for scale in scales}, "confirmation": {scale: [] for scale in scales}}
    metadata = []

    for stage, seeds in (("discovery", DISCOVERY_SEEDS), ("confirmation", CONFIRMATION_SEEDS)):
        for seed in seeds:
            outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
            for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
                X_train, X_valid = X.iloc[outer_train], X.iloc[outer_valid]
                y_train = y[outer_train]
                inner_oof = np.zeros(len(outer_train), dtype=float)
                inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed * 100 + outer_fold)
                for inner_train, inner_valid in inner.split(X_train):
                    preprocessor = phase2.onehot_preprocessor(X_train.iloc[inner_train])
                    train_matrix = preprocessor.fit_transform(X_train.iloc[inner_train])
                    valid_matrix = preprocessor.transform(X_train.iloc[inner_valid])
                    base_model = clone(model_spec["model"]).fit(train_matrix, y_train[inner_train])
                    inner_oof[inner_valid] = base_model.predict(valid_matrix)

                outer_preprocessor = phase2.onehot_preprocessor(X_train)
                outer_train_matrix = outer_preprocessor.fit_transform(X_train)
                outer_valid_matrix = outer_preprocessor.transform(X_valid)
                outer_model = clone(model_spec["model"]).fit(outer_train_matrix, y_train)
                base_prediction = outer_model.predict(outer_valid_matrix)
                residual_target = y_train - inner_oof

                train_cat = catboost_frame(X_train, categorical_columns)
                valid_cat = catboost_frame(X_valid, categorical_columns)
                residual_model = CatBoostRegressor(
                    **CATBOOST_PARAMS,
                    random_seed=seed * 100 + outer_fold,
                )
                residual_model.fit(train_cat, residual_target, cat_features=categorical_columns)
                correction = residual_model.predict(valid_cat)
                actual = y[outer_valid]
                champion_prediction = champion_log[outer_valid]
                actual_by_stage[stage].extend(actual.tolist())
                base_by_stage[stage].extend(champion_prediction.tolist())
                corrections[stage].extend(correction.tolist())
                for scale in scales:
                    fold_scores[stage][scale].append(
                        rmse(actual, champion_prediction + scale * correction)
                    )
                metadata.extend(
                    {
                        "Id": int(train.iloc[row_index].Id),
                        "stage": stage,
                        "seed": seed,
                        "fold": outer_fold,
                        "y_log": float(y[row_index]),
                        "champion_oof_log": float(champion_log[row_index]),
                        "catboost_residual_correction": float(correction[offset]),
                    }
                    for offset, row_index in enumerate(outer_valid)
                )
                print(f"{stage} seed={seed} fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    discovery_scores = {}
    confirmation_scores = {}
    for scale in scales:
        discovery_prediction = np.asarray(base_by_stage["discovery"]) + scale * np.asarray(corrections["discovery"])
        confirmation_prediction = np.asarray(base_by_stage["confirmation"]) + scale * np.asarray(corrections["confirmation"])
        discovery_scores[scale] = summarize(
            actual_by_stage["discovery"],
            discovery_prediction.tolist(),
            fold_scores["discovery"][scale],
        )
        confirmation_scores[scale] = summarize(
            actual_by_stage["confirmation"],
            confirmation_prediction.tolist(),
            fold_scores["confirmation"][scale],
        )
    selected_scale = min(scales, key=lambda scale: discovery_scores[scale]["pooled_rmse"])
    baseline_confirmation = confirmation_scores[0.0]
    selected_confirmation = confirmation_scores[selected_scale]
    gain = baseline_confirmation["pooled_rmse"] - selected_confirmation["pooled_rmse"]
    worst_change = selected_confirmation["worst_fold_rmse"] - baseline_confirmation["worst_fold_rmse"]
    per_seed = {}
    correction_array = np.asarray(corrections["confirmation"])
    base_array = np.asarray(base_by_stage["confirmation"])
    actual_array = np.asarray(actual_by_stage["confirmation"])
    offset = 0
    for seed in CONFIRMATION_SEEDS:
        size = len(train)
        per_seed[str(seed)] = {
            "base_rmse": rmse(actual_array[offset : offset + size], base_array[offset : offset + size]),
            "candidate_rmse": rmse(
                actual_array[offset : offset + size],
                base_array[offset : offset + size] + selected_scale * correction_array[offset : offset + size],
            ),
        }
        offset += size
    seed_wins = sum(value["candidate_rmse"] < value["base_rmse"] for value in per_seed.values())
    eligible = gain >= 0.0003 and worst_change <= 0.01 and seed_wins >= 2

    oof_path = EXPERIMENTS / "phase6_catboost_residual_oof.csv"
    report_path = EXPERIMENTS / "phase6_catboost_residual_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase6_catboost_residual.csv"
    components_path = EXPERIMENTS / "phase6_catboost_residual_test_components.csv"
    if oof_path.exists() or report_path.exists() or (eligible and (candidate_path.exists() or components_path.exists())):
        raise FileExistsError("Refusing to overwrite phase-six CatBoost residual artifacts")
    oof = pd.DataFrame(metadata)
    selected_oof = oof.loc[oof.stage.eq("discovery") | oof.stage.eq("confirmation")].copy()
    selected_oof["candidate_log"] = selected_oof.champion_oof_log + selected_scale * selected_oof.catboost_residual_correction
    selected_oof.to_csv(oof_path, index=False)

    candidate_written = False
    if eligible:
        inner_oof_full = np.zeros(len(X), dtype=float)
        inner_full = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2042)
        for inner_train, inner_valid in inner_full.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"]).fit(train_matrix, y[inner_train])
            inner_oof_full[inner_valid] = model.predict(valid_matrix)
        residual_full = y - inner_oof_full
        train_cat = catboost_frame(X, categorical_columns)
        test_cat = catboost_frame(X_test, categorical_columns)
        residual_model = CatBoostRegressor(**CATBOOST_PARAMS, random_seed=2042)
        residual_model.fit(train_cat, residual_full, cat_features=categorical_columns)
        correction_test = residual_model.predict(test_cat)
        champion = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
        candidate_price = np.expm1(
            np.log1p(champion.SalePrice.to_numpy(dtype=float)) + selected_scale * correction_test
        )
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("CatBoost residual candidate contains invalid prices")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
        pd.DataFrame(
            {
                "Id": test.Id,
                "champion_price": champion.SalePrice,
                "catboost_residual_correction_log": correction_test,
                "scale": selected_scale,
                "candidate_price": candidate_price,
            }
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P6-NATIVE-CATBOOST-RESIDUAL",
        "hypothesis": "Native-categorical CatBoost can model nonlinear feature interactions in the champion residuals beyond the XGBoost residual hierarchy.",
        "residual_firewall": "Outer training residual targets are built from inner-OOF XGBoost predictions; outer validation rows do not contribute to residual fit.",
        "residual_model": {**CATBOOST_PARAMS, "random_seed": "seed×fold+fold", "categorical_features": "all non-numeric columns"},
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "selected_scale_on_discovery_only": selected_scale,
        "discovery_metrics_by_scale": {str(scale): value for scale, value in discovery_scores.items()},
        "baseline_confirmation": baseline_confirmation,
        "candidate_confirmation": selected_confirmation,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": per_seed,
        "confirmation_seeds_improved": seed_wins,
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "components_path": str(components_path.relative_to(ROOT)) if candidate_written else None,
        "current_external_champion": 0.12374,
        "external_score": None,
        "decision": "SHADOW_ONLY" if eligible else "REJECT",
        "caveat": "Candidate evaluated by transfer to champion's fixed crossfit OOF; not repeated retraining of the full champion.",
        "oof_path": str(oof_path.relative_to(ROOT)),
        "runtime_seconds": round(time.perf_counter() - started, 2),
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
                "hypothesis": report["hypothesis"],
                "change": "Native CatBoost correction on inner-OOF XGBoost residuals",
                "cv": f"{selected_confirmation['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirmation['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested KFold5 outer/KFold4 inner; discovery/confirmation 3 seeds each",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected residual scale={selected_scale:g}; gain={gain:.6f}; current champion OOF transfer",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()