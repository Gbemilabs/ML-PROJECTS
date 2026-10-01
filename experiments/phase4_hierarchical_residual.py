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

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (13, 31, 73)
CONFIRMATION_SEEDS = (2025, 2026, 2027)
OUTER_SPLITS = 5
INNER_SPLITS = 4
SHRINKAGES = (5.0, 20.0, 60.0)
GROUPS = (
    "Neighborhood",
    "OverallQual",
    "SaleCondition",
    "SaleType_x_OverallQual",
    "Neighborhood_x_OverallQual",
    "Neighborhood_x_SaleCondition",
    "Neighborhood_x_OverallQual_x_SaleCondition",
    "OverallQual_x_GrLivAreaQuartile",
)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def normalized(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna("Missing").astype(str).to_numpy()


def group_keys(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fit_neighborhood = normalized(fit_frame, "Neighborhood")
    query_neighborhood = normalized(query_frame, "Neighborhood")
    fit_quality = normalized(fit_frame, "OverallQual")
    query_quality = normalized(query_frame, "OverallQual")
    fit_sale_type = normalized(fit_frame, "SaleType")
    query_sale_type = normalized(query_frame, "SaleType")
    fit_condition = normalized(fit_frame, "SaleCondition")
    query_condition = normalized(query_frame, "SaleCondition")

    quartiles = np.unique(
        np.quantile(fit_frame["GrLivArea"].to_numpy(dtype=float), [0.25, 0.5, 0.75])
    )
    bins = np.r_[-np.inf, quartiles, np.inf]
    fit_size = pd.cut(fit_frame["GrLivArea"], bins=bins, labels=False).fillna(-1).astype(str).to_numpy()
    query_size = pd.cut(query_frame["GrLivArea"], bins=bins, labels=False).fillna(-1).astype(str).to_numpy()

    fit_keys = {
        "Neighborhood": fit_neighborhood,
        "OverallQual": fit_quality,
        "SaleCondition": fit_condition,
        "SaleType_x_OverallQual": np.char.add(np.char.add(fit_sale_type, "|"), fit_quality),
        "Neighborhood_x_OverallQual": np.char.add(np.char.add(fit_neighborhood, "|"), fit_quality),
        "Neighborhood_x_SaleCondition": np.char.add(np.char.add(fit_neighborhood, "|"), fit_condition),
        "Neighborhood_x_OverallQual_x_SaleCondition": np.char.add(
            np.char.add(np.char.add(np.char.add(fit_neighborhood, "|"), fit_quality), "|"),
            fit_condition,
        ),
        "OverallQual_x_GrLivAreaQuartile": np.char.add(np.char.add(fit_quality, "|"), fit_size),
    }
    query_keys = {
        "Neighborhood": query_neighborhood,
        "OverallQual": query_quality,
        "SaleCondition": query_condition,
        "SaleType_x_OverallQual": np.char.add(np.char.add(query_sale_type, "|"), query_quality),
        "Neighborhood_x_OverallQual": np.char.add(np.char.add(query_neighborhood, "|"), query_quality),
        "Neighborhood_x_SaleCondition": np.char.add(np.char.add(query_neighborhood, "|"), query_condition),
        "Neighborhood_x_OverallQual_x_SaleCondition": np.char.add(
            np.char.add(np.char.add(np.char.add(query_neighborhood, "|"), query_quality), "|"),
            query_condition,
        ),
        "OverallQual_x_GrLivAreaQuartile": np.char.add(np.char.add(query_quality, "|"), query_size),
    }
    return fit_keys, query_keys


def group_residual_correction(
    fit_keys: np.ndarray,
    query_keys: np.ndarray,
    residuals: np.ndarray,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    stats = pd.DataFrame({"key": fit_keys, "residual": residuals}).groupby("key").residual.agg(
        ["mean", "count"]
    )
    means = pd.Series(query_keys).map(stats["mean"]).fillna(0.0).to_numpy(dtype=float)
    counts = pd.Series(query_keys).map(stats["count"]).fillna(0.0).to_numpy(dtype=float)
    correction = counts / (counts + shrinkage) * means
    return correction, counts


def summarize(
    actual: list[float],
    predictions: list[float],
    fold_scores: list[float],
) -> dict[str, float]:
    error = np.asarray(predictions) - np.asarray(actual)
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
    configs = [(group_name, shrinkage) for group_name in GROUPS for shrinkage in SHRINKAGES]
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
    baseline_fold_scores = {"discovery": [], "confirmation": []}
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
                    inner_model = clone(spec["model"])
                    inner_model.fit(train_matrix, y_train[inner_train])
                    inner_oof[inner_valid] = inner_model.predict(valid_matrix)

                outer_preprocessor = phase2.onehot_preprocessor(X_train)
                outer_train_matrix = outer_preprocessor.fit_transform(X_train)
                outer_valid_matrix = outer_preprocessor.transform(X_valid)
                outer_model = clone(spec["model"])
                outer_model.fit(outer_train_matrix, y_train)
                base = outer_model.predict(outer_valid_matrix)
                residual = y_train - inner_oof
                fit_keys, valid_keys = group_keys(raw_fold_train, raw_fold_valid)
                actual = y[outer_valid]
                actual_by_stage[stage].extend(actual.tolist())
                base_by_stage[stage].extend(base.tolist())
                baseline_fold_scores[stage].append(rmse(actual, base))

                for config in configs:
                    group_name, shrinkage = config
                    correction, counts = group_residual_correction(
                        fit_keys[group_name],
                        valid_keys[group_name],
                        residual,
                        shrinkage,
                    )
                    corrected = base + correction
                    predictions[stage][config].extend(corrected.tolist())
                    fold_scores[stage][config].append(rmse(actual, corrected))

                metadata.extend(
                    {
                        "Id": int(raw_train.iloc[index]["Id"]),
                        "stage": stage,
                        "seed": seed,
                        "fold": outer_fold,
                        "y_log": float(y[index]),
                        "xgb_base_log": float(base[offset]),
                    }
                    for offset, index in enumerate(outer_valid)
                )
                print(f"{stage} seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    baseline_summary = {
        stage: {
            "pooled_rmse": rmse(actual_by_stage[stage], base_by_stage[stage]),
            "mean_fold_rmse": float(np.mean(baseline_fold_scores[stage])),
            "fold_rmse_std": float(np.std(baseline_fold_scores[stage])),
            "worst_fold_rmse": float(np.max(baseline_fold_scores[stage])),
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
    selected_config = min(discovery_metrics, key=lambda config: discovery_metrics[config]["pooled_rmse"])
    confirmation_metrics = {
        config: summarize(
            actual_by_stage["confirmation"],
            predictions["confirmation"][config],
            fold_scores["confirmation"][config],
        )
        for config in configs
    }
    selected_confirmation = confirmation_metrics[selected_config]
    pooled_gain = baseline_summary["confirmation"]["pooled_rmse"] - selected_confirmation["pooled_rmse"]
    worst_fold_change = (
        selected_confirmation["worst_fold_rmse"]
        - baseline_summary["confirmation"]["worst_fold_rmse"]
    )
    eligible = pooled_gain >= 0.0003 and worst_fold_change <= 0.01

    oof_path = EXPERIMENTS / "hierarchical_residual_oof.csv"
    report_path = EXPERIMENTS / "hierarchical_residual_validation.json"
    shadow_dir = EXPERIMENTS / "shadow_candidates"
    candidate_path = shadow_dir / "hierarchical_residual_xgb.csv"
    if oof_path.exists() or report_path.exists() or (eligible and candidate_path.exists()):
        raise FileExistsError("Refusing to overwrite existing hierarchical-residual artifacts")

    selected_values = (
        predictions["discovery"][selected_config]
        + predictions["confirmation"][selected_config]
    )
    oof_frame = pd.DataFrame(metadata)
    oof_frame["selected_group_residual_log"] = selected_values
    oof_frame.to_csv(oof_path, index=False)

    candidate_written = False
    if eligible:
        full_preprocessor = phase2.onehot_preprocessor(X)
        full_train_matrix = full_preprocessor.fit_transform(X)
        test_matrix = full_preprocessor.transform(X_test)
        full_model = clone(spec["model"])
        full_model.fit(full_train_matrix, y)
        base_test = full_model.predict(test_matrix)
        inner_oof_full = np.zeros(len(X), dtype=float)
        inner_full = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2030)
        for inner_train, inner_valid in inner_full.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(spec["model"])
            model.fit(train_matrix, y[inner_train])
            inner_oof_full[inner_valid] = model.predict(valid_matrix)
        full_residual = y - inner_oof_full
        train_keys, test_keys = group_keys(raw_train, raw_test)
        correction, _ = group_residual_correction(
            train_keys[selected_config[0]],
            test_keys[selected_config[0]],
            full_residual,
            selected_config[1],
        )
        candidate_price = np.expm1(base_test + correction)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Hierarchical residual shadow candidate has invalid prices")
        shadow_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": raw_test["Id"], "SalePrice": candidate_price}).to_csv(
            candidate_path,
            index=False,
        )
        candidate_written = True

    ranked = sorted(
        (
            {
                "group": group_name,
                "shrinkage": shrinkage,
                **discovery_metrics[(group_name, shrinkage)],
                "confirmation": confirmation_metrics[(group_name, shrinkage)],
            }
            for group_name, shrinkage in configs
        ),
        key=lambda row: row["pooled_rmse"],
    )
    report = {
        "experiment_id": "P4-HIERARCHICAL-RESIDUAL",
        "hypothesis": "Fold-local residual shrinkage by market and property segments may capture systematic local price formation beyond a global regressor.",
        "residual_target_firewall": "For each outer fold, residual targets are generated by inner OOF XGBoost predictions using only outer-training rows.",
        "base_model": "XGBoost_regularized; phase-two fixed parameters",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "group_families": list(GROUPS),
        "shrinkage_values": list(SHRINKAGES),
        "selected_from_discovery_only": {
            "group": selected_config[0],
            "shrinkage": selected_config[1],
        },
        "baseline": baseline_summary,
        "selected_discovery": discovery_metrics[selected_config],
        "selected_confirmation": selected_confirmation,
        "confirmation_pooled_gain": pooled_gain,
        "confirmation_worst_fold_change": worst_fold_change,
        "confirmation_candidate_ranking": ranked,
        "shadow_gate": "Confirmation pooled RMSE gain >= 0.0003 and worst fold <= baseline worst + 0.01",
        "eligible_for_shadow_candidate": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "external_champion_score": 0.12654,
        "external_champion_nested_oof_available": False,
        "external_score": None,
        "oof_path": str(oof_path.relative_to(ROOT)),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "decision": "SHADOW_ONLY_LOCAL_XGB" if eligible else "REJECT",
        "caveat": "This challenges XGBoost alone, not the protected 70/30 external champion; no Kaggle promotion is implied.",
    }
    report_path.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in log_rows):
        raise ValueError("Hierarchical-residual experiment already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Nested group-residual offsets for neighborhood/quality/sale/size regimes",
                "cv": f"{selected_confirmation['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirmation['fold_rmse_std']:.6f}",
                "validation_scheme": "Discovery/confirmation nested KFold5 outer / KFold4 inner; three seeds each",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected {selected_config[0]} k={selected_config[1]:g}; pooled gain={pooled_gain:.6f}; worst-fold change={worst_fold_change:+.6f}; XGBoost-only",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()