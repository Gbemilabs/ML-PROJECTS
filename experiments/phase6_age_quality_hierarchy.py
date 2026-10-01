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
DISCOVERY_SEEDS = (2021, 2022)
CONFIRMATION_SEEDS = (2023, 2024, 2025)
OUTER_SPLITS = 5
INNER_SPLITS = 4
REPRESENTATIONS = ("base_hierarchy", "quality_age", "quality_remodel", "neighborhood_age", "all_age_interactions")
MIN_FREQUENCIES = (10, 20)
RIDGE_ALPHAS = (10.0, 50.0)
SCALES = (0.25, 0.5, 0.75)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


def summary(actual: list[float], prediction: list[float], fold_rmse: list[float]) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(actual)
    return {
        "pooled_rmse": float(np.mean(error * error) ** 0.5),
        "mean_fold_rmse": float(np.mean(fold_rmse)),
        "fold_rmse_std": float(np.std(fold_rmse)),
        "worst_fold_rmse": float(np.max(fold_rmse)),
        "median_fold_rmse": float(np.median(fold_rmse)),
    }


def _category(series: pd.Series) -> np.ndarray:
    return series.fillna("Missing").astype(str).to_numpy()


def age_feature_frames(
    fit_raw: pd.DataFrame,
    query_raw: pd.DataFrame,
    representation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = pd.DataFrame(index=np.arange(len(fit_raw)))
    query = pd.DataFrame(index=np.arange(len(query_raw)))
    fit_base = {
        name: _category(fit_raw[name])
        for name in ("Neighborhood", "OverallQual", "OverallCond", "SaleType", "SaleCondition", "YrSold")
    }
    query_base = {
        name: _category(query_raw[name])
        for name in fit_base
    }
    for name in fit_base:
        fit[name] = fit_base[name]
        query[name] = query_base[name]
    fit_neighborhood, query_neighborhood = fit_base["Neighborhood"], query_base["Neighborhood"]
    fit_quality, query_quality = fit_base["OverallQual"], query_base["OverallQual"]

    fit_house_age = (fit_raw.YrSold - fit_raw.YearBuilt).to_numpy(dtype=float)
    query_house_age = (query_raw.YrSold - query_raw.YearBuilt).to_numpy(dtype=float)
    fit_remodel_age = (fit_raw.YrSold - fit_raw.YearRemodAdd).to_numpy(dtype=float)
    query_remodel_age = (query_raw.YrSold - query_raw.YearRemodAdd).to_numpy(dtype=float)
    age_edges = np.unique(np.quantile(fit_house_age, [0.25, 0.5, 0.75]))
    remodel_edges = np.unique(np.quantile(fit_remodel_age, [0.25, 0.5, 0.75]))
    fit_age_bin = pd.cut(fit_house_age, bins=np.r_[-np.inf, age_edges, np.inf], labels=False).astype(str)
    query_age_bin = pd.cut(query_house_age, bins=np.r_[-np.inf, age_edges, np.inf], labels=False).astype(str)
    fit_remodel_bin = pd.cut(
        fit_remodel_age,
        bins=np.r_[-np.inf, remodel_edges, np.inf],
        labels=False,
    ).astype(str)
    query_remodel_bin = pd.cut(
        query_remodel_age,
        bins=np.r_[-np.inf, remodel_edges, np.inf],
        labels=False,
    ).astype(str)
    fit["Neighborhood_x_Quality"] = fit_neighborhood + "|" + fit_quality
    query["Neighborhood_x_Quality"] = query_neighborhood + "|" + query_quality
    if representation in {"quality_age", "all_age_interactions"}:
        fit["Quality_x_HouseAge"] = fit_quality + "|" + fit_age_bin
        query["Quality_x_HouseAge"] = query_quality + "|" + query_age_bin
    if representation in {"quality_remodel", "all_age_interactions"}:
        fit["Quality_x_RemodelAge"] = fit_quality + "|" + fit_remodel_bin
        query["Quality_x_RemodelAge"] = query_quality + "|" + query_remodel_bin
    if representation in {"neighborhood_age", "all_age_interactions"}:
        fit["Neighborhood_x_HouseAge"] = fit_neighborhood + "|" + fit_age_bin
        query["Neighborhood_x_HouseAge"] = query_neighborhood + "|" + query_age_bin
    return fit.astype(str), query.astype(str)


def fit_correction(
    fit_groups: pd.DataFrame,
    valid_groups: pd.DataFrame,
    residual: np.ndarray,
    alpha: float,
    min_frequency: int,
) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=min_frequency, sparse_output=True)
    fit_matrix = encoder.fit_transform(fit_groups)
    valid_matrix = encoder.transform(valid_groups)
    model = Ridge(alpha=alpha).fit(fit_matrix, residual)
    return model.predict(valid_matrix)


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
    if champion_oof.Id.duplicated().any() or set(champion_oof.Id) != set(train.Id):
        raise ValueError("Protected champion OOF IDs are incomplete or duplicated")
    champion_oof = champion_oof.set_index("Id").reindex(train.Id).reset_index()
    if not np.array_equal(champion_submission.Id, test.Id):
        raise ValueError("Protected champion submission IDs differ from test IDs")

    X, _, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    base_log = champion_oof.candidate_oof_log.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]
    configs = [
        (representation, alpha, min_frequency, scale)
        for representation in REPRESENTATIONS
        for alpha in RIDGE_ALPHAS
        for min_frequency in MIN_FREQUENCIES
        for scale in SCALES
    ]
    actual_by_stage = {"discovery": [], "confirmation": []}
    corrections_by_stage = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    folds_by_stage = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    champion_by_stage = {"discovery": [], "confirmation": []}
    metadata = []

    for stage, seeds in (("discovery", DISCOVERY_SEEDS), ("confirmation", CONFIRMATION_SEEDS)):
        for seed in seeds:
            outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
            for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
                X_train, X_valid = X.iloc[outer_train], X.iloc[outer_valid]
                raw_train, raw_valid = train.iloc[outer_train], train.iloc[outer_valid]
                y_train = y[outer_train]
                inner_oof = np.zeros(len(outer_train), dtype=float)
                inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed * 100 + outer_fold)
                for inner_train, inner_valid in inner.split(X_train):
                    preprocessor = phase2.onehot_preprocessor(X_train.iloc[inner_train])
                    train_matrix = preprocessor.fit_transform(X_train.iloc[inner_train])
                    valid_matrix = preprocessor.transform(X_train.iloc[inner_valid])
                    model = clone(model_spec["model"]).fit(train_matrix, y_train[inner_train])
                    inner_oof[inner_valid] = model.predict(valid_matrix)
                residual = y_train - inner_oof
                actual = y[outer_valid]
                actual_by_stage[stage].extend(actual.tolist())
                champion_by_stage[stage].extend(base_log[outer_valid].tolist())
                metadata.append(
                    pd.DataFrame(
                        {
                            "Id": train.iloc[outer_valid].Id.to_numpy(dtype=int),
                            "stage": stage,
                            "seed": seed,
                            "fold": outer_fold,
                            "y_log": actual,
                            "current_champion_oof_log": base_log[outer_valid],
                        }
                    )
                )
                group_cache = {
                    representation: age_feature_frames(raw_train, raw_valid, representation)
                    for representation in REPRESENTATIONS
                }
                for representation, alpha, min_frequency, scale in configs:
                    fit_groups, valid_groups = group_cache[representation]
                    correction = fit_correction(fit_groups, valid_groups, residual, alpha, min_frequency)
                    corrections_by_stage[stage][(representation, alpha, min_frequency, scale)].extend(
                        correction.tolist()
                    )
                    folds_by_stage[stage][(representation, alpha, min_frequency, scale)].append(
                        rmse(actual, base_log[outer_valid] + scale * correction)
                    )
                print(f"{stage} seed={seed} fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    baseline_by_stage = {
        stage: {
            "pooled_rmse": rmse(actual_by_stage[stage], champion_by_stage[stage]),
            "folds": [],
        }
        for stage in ("discovery", "confirmation")
    }
    for stage in ("discovery", "confirmation"):
        values = []
        cursor = 0
        for _seed in (DISCOVERY_SEEDS if stage == "discovery" else CONFIRMATION_SEEDS):
            for _fold in range(OUTER_SPLITS):
                fold_size = len(train) // OUTER_SPLITS
                actual = np.asarray(actual_by_stage[stage])[cursor : cursor + fold_size]
                base_prediction = np.asarray(champion_by_stage[stage])[cursor : cursor + fold_size]
                values.append(rmse(actual, base_prediction))
                cursor += fold_size
        baseline_by_stage[stage]["folds"] = values

    discovery_scores = {}
    confirmation_scores = {}
    for config in configs:
        for stage, target in (("discovery", discovery_scores), ("confirmation", confirmation_scores)):
            pred = np.asarray(champion_by_stage[stage]) + np.asarray(corrections_by_stage[stage][config])
            target[config] = summary(actual_by_stage[stage], pred.tolist(), folds_by_stage[stage][config])
    selected = min(configs, key=lambda config: discovery_scores[config]["pooled_rmse"])
    selected_confirm = confirmation_scores[selected]
    confirmation_base = summary(
        actual_by_stage["confirmation"],
        champion_by_stage["confirmation"],
        baseline_by_stage["confirmation"]["folds"],
    )
    gain = confirmation_base["pooled_rmse"] - selected_confirm["pooled_rmse"]
    worst_change = selected_confirm["worst_fold_rmse"] - confirmation_base["worst_fold_rmse"]
    per_seed = {}
    selected_correction = np.asarray(corrections_by_stage["confirmation"][selected])
    offset = 0
    for seed in CONFIRMATION_SEEDS:
        n = len(train)
        actual = np.asarray(actual_by_stage["confirmation"])[offset : offset + n]
        base_prediction = np.asarray(champion_by_stage["confirmation"])[offset : offset + n]
        correction = selected_correction[offset : offset + n]
        per_seed[str(seed)] = {
            "base_rmse": rmse(actual, base_prediction),
            "candidate_rmse": rmse(actual, base_prediction + correction),
        }
        offset += n
    seed_wins = sum(value["candidate_rmse"] < value["base_rmse"] for value in per_seed.values())
    eligible = gain >= 0.0003 and worst_change <= 0.01 and seed_wins >= 2

    out_oof = EXPERIMENTS / "phase6_age_quality_hierarchy_oof.csv"
    out_report = EXPERIMENTS / "phase6_age_quality_hierarchy_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase6_age_quality_hierarchy.csv"
    components_path = EXPERIMENTS / "phase6_age_quality_hierarchy_test_components.csv"
    if out_oof.exists() or out_report.exists() or (eligible and (candidate_path.exists() or components_path.exists())):
        raise FileExistsError("Refusing to overwrite phase-six age/quality experiment artifacts")
    oof = pd.concat(metadata, ignore_index=True)
    oof["selected_correction"] = np.concatenate(
        [corrections_by_stage["discovery"][selected], corrections_by_stage["confirmation"][selected]]
    )
    oof["candidate_log"] = oof.current_champion_oof_log + oof.selected_correction
    oof.to_csv(out_oof, index=False)

    candidate_written = False
    if eligible:
        X, X_test, y_series = phase2.prepare_raw(train, test)
        y_full = y_series.to_numpy(dtype=float)
        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2037)
        for inner_train, inner_valid in inner.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            xtr = preprocessor.fit_transform(X.iloc[inner_train])
            xva = preprocessor.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"]).fit(xtr, y_full[inner_train])
            inner_oof[inner_valid] = model.predict(xva)
        fit_groups, test_groups = age_feature_frames(train, test, selected[0])
        residual = y_full - inner_oof
        alpha, min_frequency = selected[1], selected[2]
        correction = fit_correction(fit_groups, test_groups, residual, alpha, min_frequency)
        champion = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
        candidate_price = np.expm1(np.log1p(champion.SalePrice.to_numpy(dtype=float)) + correction)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Age/quality candidate contains invalid predictions")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
        pd.DataFrame(
            {"Id": test.Id, "champion_price": champion.SalePrice, "residual_correction": correction, "candidate_price": candidate_price}
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P6-AGE-QUALITY-RESIDUAL-HIERARCHY",
        "hypothesis": "Residual errors among old low-quality homes arise from a quality-dependent age/renovation valuation curve, not merely neighborhood singleton effects.",
        "features": ["quality×house-age-band", "quality×remodel-age-band", "neighborhood×house-age-band"],
        "residual_firewall": "Nested inner-OOF XGBoost residuals; group boundaries are fitted within outer training data; outer-validation targets never enter the residual fit.",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "representation_options": list(REPRESENTATIONS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "min_frequency": list(MIN_FREQUENCIES),
        "correction_scales": list(SCALES),
        "selected_on_discovery_only": {"representation": selected[0], "alpha": selected[1], "min_frequency": selected[2], "scale": selected[3]},
        "baseline_confirmation": confirmation_base,
        "candidate_confirmation": selected_confirm,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": per_seed,
        "confirmation_seeds_improved": seed_wins,
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "protected_external_champion": 0.12374,
        "external_score": None,
        "decision": "SHADOW_ONLY" if eligible else "REJECT",
        "caveat": "Champion base OOF is one saved crossfit; this tests transfer of a nested XGB residual correction, not repeated refits of the full CatBoost/XGBoost champion.",
        "oof_path": str(out_oof.relative_to(ROOT)),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    out_report.write_text(json.dumps(report, indent=2))

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
                "change": "Age/quality/renovation nested residual hierarchy transfer",
                "cv": f"{selected_confirm['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirm['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested KFold5 outer/KFold4 inner; discovery 3 seeds, confirmation 3 seeds",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected {selected[0]}, alpha={selected[1]:g}, minfreq={selected[2]}, scale={selected[3]:g}; confirmation gain={gain:.6f}",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()