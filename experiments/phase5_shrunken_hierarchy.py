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

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (19, 43, 89)
CONFIRMATION_SEEDS = (2024, 2025, 2029)
OUTER_SPLITS = 5
INNER_SPLITS = 4
ALPHAS = (10.0, 50.0, 200.0)
MIN_FREQUENCIES = (2, 5)
REPRESENTATIONS = ("parents", "hierarchy", "hierarchy_plus_size")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def config_list() -> list[tuple[str, float, int]]:
    return [
        (representation, alpha, minimum_frequency)
        for representation in REPRESENTATIONS
        for alpha in ALPHAS
        for minimum_frequency in MIN_FREQUENCIES
    ]


def group_frame(
    fit_raw: pd.DataFrame,
    query_raw: pd.DataFrame,
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = pd.DataFrame(index=np.arange(len(fit_raw)))
    query = pd.DataFrame(index=np.arange(len(query_raw)))
    values = {
        name: (
            fit_raw[name].fillna("Missing").astype(str).to_numpy(),
            query_raw[name].fillna("Missing").astype(str).to_numpy(),
        )
        for name in ("Neighborhood", "OverallQual", "OverallCond", "YrSold", "SaleType", "SaleCondition")
    }
    for name in ("Neighborhood", "OverallQual", "OverallCond", "YrSold", "SaleType", "SaleCondition"):
        fit[name], query[name] = values[name]
    fit_neighborhood, query_neighborhood = values["Neighborhood"]
    fit_quality, query_quality = values["OverallQual"]
    fit_condition, query_condition = values["SaleCondition"]
    fit_sale_type, query_sale_type = values["SaleType"]

    if representation in {"hierarchy", "hierarchy_plus_size"}:
        fit["Neighborhood_x_Quality"] = fit_neighborhood + "|" + fit_quality
        query["Neighborhood_x_Quality"] = query_neighborhood + "|" + query_quality
        fit["Neighborhood_x_Quality_x_SaleCondition"] = (
            fit_neighborhood + "|" + fit_quality + "|" + fit_condition
        )
        query["Neighborhood_x_Quality_x_SaleCondition"] = (
            query_neighborhood + "|" + query_quality + "|" + query_condition
        )
        fit["SaleType_x_Quality"] = fit_sale_type + "|" + fit_quality
        query["SaleType_x_Quality"] = query_sale_type + "|" + query_quality
    if representation == "hierarchy_plus_size":
        living = fit_raw["GrLivArea"].to_numpy(dtype=float)
        edges = np.unique(np.quantile(living, [0.25, 0.5, 0.75]))
        bins = np.r_[-np.inf, edges, np.inf]
        fit_size = pd.cut(living, bins=bins, labels=False, include_lowest=True).astype(str)
        query_size = pd.cut(
            query_raw["GrLivArea"].to_numpy(dtype=float),
            bins=bins,
            labels=False,
            include_lowest=True,
        ).astype(str)
        fit["Quality_x_AreaQuartile"] = fit_quality + "|" + fit_size
        query["Quality_x_AreaQuartile"] = query_quality + "|" + query_size
        fit["Neighborhood_x_AreaQuartile"] = fit_neighborhood + "|" + fit_size
        query["Neighborhood_x_AreaQuartile"] = query_neighborhood + "|" + query_size
    return fit.astype(str), query.astype(str)


def make_correction(
    fit_groups: pd.DataFrame,
    query_groups: pd.DataFrame,
    residual_target: np.ndarray,
    alpha: float,
    minimum_frequency: int,
) -> np.ndarray:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=minimum_frequency,
        sparse_output=True,
    )
    fit_matrix = encoder.fit_transform(fit_groups)
    query_matrix = encoder.transform(query_groups)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(fit_matrix, residual_target)
    return model.predict(query_matrix)


def summarize(
    actual: list[float],
    predicted: list[float],
    fold_scores: list[float],
) -> dict[str, float]:
    error = np.asarray(predicted) - np.asarray(actual)
    return {
        "pooled_rmse": float(np.mean(error * error) ** 0.5),
        "mean_fold_rmse": float(np.mean(fold_scores)),
        "fold_rmse_std": float(np.std(fold_scores)),
        "worst_fold_rmse": float(np.max(fold_scores)),
        "median_fold_rmse": float(np.median(fold_scores)),
    }


def main() -> None:
    started = time.perf_counter()
    raw_train = pd.read_csv(ROOT / "data" / "train.csv")
    raw_test = pd.read_csv(ROOT / "data" / "test.csv")
    X, X_test, y_series = phase2.prepare_raw(raw_train, raw_test)
    y = y_series.to_numpy(dtype=float)
    spec = phase2.model_specs()["XGBoost_regularized"]
    configs = config_list()
    actual_by_stage = {"discovery": [], "confirmation": []}
    base_by_stage = {"discovery": [], "confirmation": []}
    predictions = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    fold_scores = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    base_fold_scores = {"discovery": [], "confirmation": []}
    metadata = []

    for stage, seeds in (("discovery", DISCOVERY_SEEDS), ("confirmation", CONFIRMATION_SEEDS)):
        for seed in seeds:
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
                base_prediction = outer_model.predict(outer_valid_matrix)
                residual_target = y_train - inner_oof
                actual = y[outer_valid]

                actual_by_stage[stage].extend(actual.tolist())
                base_by_stage[stage].extend(base_prediction.tolist())
                base_fold_scores[stage].append(rmse(actual, base_prediction))
                fold_metadata = {
                    "Id": raw_train.iloc[outer_valid]["Id"].to_numpy(dtype=int),
                    "stage": stage,
                    "seed": seed,
                    "fold": outer_fold,
                    "y_log": actual,
                    "xgb_base_log": base_prediction,
                }
                metadata.append(pd.DataFrame(fold_metadata))

                for representation in REPRESENTATIONS:
                    fit_groups, valid_groups = group_frame(
                        raw_fold_train,
                        raw_fold_valid,
                        representation,
                    )
                    for alpha in ALPHAS:
                        for minimum_frequency in MIN_FREQUENCIES:
                            config = (representation, alpha, minimum_frequency)
                            correction = make_correction(
                                fit_groups,
                                valid_groups,
                                residual_target,
                                alpha,
                                minimum_frequency,
                            )
                            prediction = base_prediction + correction
                            predictions[stage][config].extend(prediction.tolist())
                            fold_scores[stage][config].append(rmse(actual, prediction))
                print(f"{stage} seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    baseline_metrics = {
        stage: {
            "pooled_rmse": rmse(actual_by_stage[stage], base_by_stage[stage]),
            "mean_fold_rmse": float(np.mean(base_fold_scores[stage])),
            "fold_rmse_std": float(np.std(base_fold_scores[stage])),
            "worst_fold_rmse": float(np.max(base_fold_scores[stage])),
        }
        for stage in ("discovery", "confirmation")
    }
    discovery_metrics = {
        config: summarize(
            actual_by_stage["discovery"],
            predictions["discovery"][config],
            fold_scores["discovery"][config],
        )
        for config in configs
    }
    selected = min(configs, key=lambda config: discovery_metrics[config]["pooled_rmse"])
    confirmation_metrics = {
        config: summarize(
            actual_by_stage["confirmation"],
            predictions["confirmation"][config],
            fold_scores["confirmation"][config],
        )
        for config in configs
    }
    selected_metrics = confirmation_metrics[selected]
    gain = baseline_metrics["confirmation"]["pooled_rmse"] - selected_metrics["pooled_rmse"]
    worst_change = selected_metrics["worst_fold_rmse"] - baseline_metrics["confirmation"]["worst_fold_rmse"]
    confirm_frames = pd.concat(metadata, ignore_index=True)
    confirm_frames = confirm_frames.loc[confirm_frames.stage.eq("confirmation")].reset_index(drop=True)
    # Per-seed challenger scores are read from the contiguous confirmation-fold prediction order.
    selected_confirmation = np.asarray(predictions["confirmation"][selected])
    offset = 0
    per_seed = {}
    for seed in CONFIRMATION_SEEDS:
        seed_count = len(raw_train)
        seed_prediction = selected_confirmation[offset : offset + seed_count]
        seed_actual = np.asarray(actual_by_stage["confirmation"])[offset : offset + seed_count]
        offset += seed_count
        per_seed[str(seed)] = {
            "base_rmse": rmse(
                confirm_frames.loc[confirm_frames.seed.eq(seed), "y_log"].to_numpy(),
                confirm_frames.loc[confirm_frames.seed.eq(seed), "xgb_base_log"].to_numpy(),
            ),
            "corrected_rmse": rmse(seed_actual, seed_prediction),
        }

    shadow_gate = gain >= 0.0003 and worst_change <= 0.01 and sum(
        values["corrected_rmse"] < values["base_rmse"] for values in per_seed.values()
    ) >= 2
    selected_name = {
        "representation": selected[0],
        "ridge_alpha": selected[1],
        "minimum_frequency": selected[2],
    }
    oof_path = EXPERIMENTS / "phase5_hierarchical_residual_oof.csv"
    report_path = EXPERIMENTS / "phase5_hierarchical_residual_validation.json"
    shadow_dir = EXPERIMENTS / "shadow_candidates"
    candidate_path = shadow_dir / "phase5_hierarchical_residual_xgb.csv"
    if oof_path.exists() or report_path.exists() or (shadow_gate and candidate_path.exists()):
        raise FileExistsError("Refusing to overwrite a previous phase-five experiment artifact")

    selected_oof = pd.concat(metadata, ignore_index=True)
    selected_oof["stage_prediction"] = (
        predictions["discovery"][selected] + predictions["confirmation"][selected]
    )
    selected_oof.to_csv(oof_path, index=False)

    candidate_written = False
    if shadow_gate:
        X_full, X_test, y_full = X, X_test, y
        full_preprocessor = phase2.onehot_preprocessor(X_full)
        full_train_matrix = full_preprocessor.fit_transform(X_full)
        test_matrix = full_preprocessor.transform(X_test)
        full_model = clone(spec["model"])
        full_model.fit(full_train_matrix, y_full)
        base_test = full_model.predict(test_matrix)

        inner_oof_full = np.zeros(len(X_full), dtype=float)
        inner_full = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2031)
        for inner_train, inner_valid in inner_full.split(X_full):
            prep = phase2.onehot_preprocessor(X_full.iloc[inner_train])
            train_matrix = prep.fit_transform(X_full.iloc[inner_train])
            valid_matrix = prep.transform(X_full.iloc[inner_valid])
            model = clone(spec["model"])
            model.fit(train_matrix, y_full[inner_train])
            inner_oof_full[inner_valid] = model.predict(valid_matrix)
        fit_groups, test_groups = group_frame(raw_train, raw_test, selected[0])
        correction = make_correction(
            fit_groups,
            test_groups,
            y_full - inner_oof_full,
            selected[1],
            selected[2],
        )
        price = np.expm1(base_test + correction)
        if not np.isfinite(price).all() or (price <= 0).any():
            raise ValueError("Phase-five shadow predictions are invalid")
        shadow_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": raw_test["Id"], "SalePrice": price}).to_csv(candidate_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P5-SHRUNK-HIERARCHICAL-RESIDUAL",
        "hypothesis": "A ridge-shrunk additive market hierarchy can capture stable residual structure while suppressing singleton cell effects.",
        "residual_target_firewall": "Every outer-training residual comes from an inner OOF XGBoost model trained without that row; outer validation labels are never used to fit encoders or residual effects.",
        "model_base": "Regularized XGBoost; phase-two fixed parameters",
        "group_features": {
            "parents": ["Neighborhood", "OverallQual", "OverallCond", "YrSold", "SaleType", "SaleCondition"],
            "interactions": ["Neighborhood×Quality", "Neighborhood×Quality×SaleCondition", "SaleType×Quality"],
            "size_features": ["Quality×AreaQuartile", "Neighborhood×AreaQuartile"],
        },
        "ridge_alphas": list(ALPHAS),
        "onehot_min_frequency": list(MIN_FREQUENCIES),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "candidate_config_count": len(configs),
        "selected_on_discovery_only": selected_name,
        "baseline": baseline_metrics,
        "selected_discovery": discovery_metrics[selected],
        "selected_confirmation": selected_metrics,
        "confirmation_pooled_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": per_seed,
        "shadow_gate": "Pooled confirmation gain >=0.0003, worst fold <=base+0.01, and at least two of three seeds improve",
        "shadow_gate_passed": bool(shadow_gate),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "protected_external_champion": 0.12654,
        "champion_matched_validation": False,
        "external_score": None,
        "oof_path": str(oof_path.relative_to(ROOT)),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "decision": "SHADOW_ONLY_XGB" if shadow_gate else "REJECT",
        "caveat": "This experiment challenges regularized XGBoost, not the protected CatBoost/XGBoost ensemble; it cannot promote root submission.csv.",
    }
    report_path.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in log_rows):
        raise ValueError("Phase-five experiment already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Ridge-shrunk categorical hierarchy on inner-OOF XGBoost residuals",
                "cv": f"{selected_metrics['pooled_rmse']:.6f}",
                "cv_std": f"{selected_metrics['fold_rmse_std']:.6f}",
                "validation_scheme": "Discovery/confirmation nested KFold5 outer / KFold4 inner; 3 seeds per stage",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected {selected_name}; confirmation gain={gain:.6f}; worst-fold change={worst_change:+.6f}; XGBoost-only, no champion-matched CV",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()