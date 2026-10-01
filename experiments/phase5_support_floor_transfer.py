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
OUTER_SPLITS = 5
INNER_SPLITS = 4
MIN_FREQUENCIES = (10, 20)
RIDGE_ALPHAS = (10.0, 50.0)
SCALES = (0.25, 0.5, 0.75, 1.0)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


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
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12654.csv")
    if not np.array_equal(champion_oof.Id, train.Id):
        raise ValueError("Champion OOF IDs do not match train rows")
    champion_log = np.log1p(champion_oof.SalePrice.to_numpy(dtype=float))
    X, X_test, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]
    configs = [(alpha, min_frequency) for alpha in RIDGE_ALPHAS for min_frequency in MIN_FREQUENCIES]
    stages = {"discovery": DISCOVERY_SEEDS, "confirmation": CONFIRMATION_SEEDS}
    records = []

    for stage, seeds in stages.items():
        for seed in seeds:
            outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
            for outer_fold, (outer_train, outer_valid) in enumerate(outer.split(X), start=1):
                X_train = X.iloc[outer_train]
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
                    model = clone(model_spec["model"])
                    model.fit(train_matrix, y_train[inner_train])
                    inner_oof[inner_valid] = model.predict(valid_matrix)
                residual = y_train - inner_oof
                fit_groups, valid_groups = hierarchy.group_frame(raw_train, raw_valid, "hierarchy")
                actual = y[outer_valid]
                for offset, row_index in enumerate(outer_valid):
                    records.append(
                        {
                            "Id": int(train.iloc[row_index].Id),
                            "stage": stage,
                            "seed": seed,
                            "fold": outer_fold,
                            "y_log": float(y[row_index]),
                            "champion_oof_log": float(champion_log[row_index]),
                        }
                    )
                predictions_for_fold = {config: [] for config in configs}
                for alpha, min_frequency in configs:
                    encoder = OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=min_frequency,
                        sparse_output=True,
                    )
                    train_matrix = encoder.fit_transform(fit_groups)
                    valid_matrix = encoder.transform(valid_groups)
                    residual_model = Ridge(alpha=alpha).fit(train_matrix, residual)
                    correction = residual_model.predict(valid_matrix)
                    predictions_for_fold[(alpha, min_frequency)] = correction
                for offset, row_index in enumerate(outer_valid):
                    record_index = len(records) - len(outer_valid) + offset
                    for config, correction in predictions_for_fold.items():
                        records[record_index][f"correction_a{config[0]:g}_m{config[1]}"] = float(
                            correction[offset]
                        )
                print(f"{stage} seed={seed} fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    results = pd.DataFrame(records)
    discovery = results.loc[results.stage.eq("discovery")].copy()
    confirmation = results.loc[results.stage.eq("confirmation")].copy()
    discovery_scores = {}
    confirmation_scores = {}
    for alpha, min_frequency in configs:
        correction_column = f"correction_a{alpha:g}_m{min_frequency}"
        for scale in SCALES:
            key = (alpha, min_frequency, scale)
            prediction_column = f"candidate_a{alpha:g}_m{min_frequency}_s{scale:g}"
            discovery[prediction_column] = (
                discovery.champion_oof_log + scale * discovery[correction_column]
            )
            confirmation[prediction_column] = (
                confirmation.champion_oof_log + scale * confirmation[correction_column]
            )
            discovery_scores[key] = summarize(discovery, prediction_column)
            confirmation_scores[key] = summarize(confirmation, prediction_column)

    selected = min(discovery_scores, key=lambda key: discovery_scores[key]["pooled_rmse"])
    alpha, min_frequency, scale = selected
    prediction_column = f"candidate_a{alpha:g}_m{min_frequency}_s{scale:g}"
    base_confirmation = summarize(confirmation, "champion_oof_log")
    selected_confirmation = confirmation_scores[selected]
    gain = base_confirmation["pooled_rmse"] - selected_confirmation["pooled_rmse"]
    worst_change = selected_confirmation["worst_fold_rmse"] - base_confirmation["worst_fold_rmse"]
    per_seed = {}
    for seed, rows in confirmation.groupby("seed"):
        per_seed[str(int(seed))] = {
            "base": rmse(rows.y_log.to_numpy(), rows.champion_oof_log.to_numpy()),
            "candidate": rmse(rows.y_log.to_numpy(), rows[prediction_column].to_numpy()),
        }
    seed_wins = sum(item["candidate"] < item["base"] for item in per_seed.values())
    eligible = gain >= 0.0003 and worst_change <= 0.01 and seed_wins >= 2

    output_oof = EXPERIMENTS / "phase5_support_floor_transfer_oof.csv"
    output_report = EXPERIMENTS / "phase5_support_floor_transfer_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase5_support_floor_transfer.csv"
    components_path = EXPERIMENTS / "phase5_support_floor_test_components.csv"
    if output_oof.exists() or output_report.exists() or (
        eligible and (candidate_path.exists() or components_path.exists())
    ):
        raise FileExistsError("Refusing to overwrite support-floor transfer artifacts")
    results.to_csv(output_oof, index=False)

    candidate_written = False
    if eligible:
        X, X_test, y_series = phase2.prepare_raw(train, test)
        y = y_series.to_numpy(dtype=float)
        full_prep = phase2.onehot_preprocessor(X)
        train_matrix = full_prep.fit_transform(X)
        test_matrix = full_prep.transform(X_test)
        full_model = clone(model_spec["model"]).fit(train_matrix, y)
        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2036)
        for inner_train, inner_valid in inner.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            inner_train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            inner_valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"]).fit(inner_train_matrix, y[inner_train])
            inner_oof[inner_valid] = model.predict(inner_valid_matrix)
        fit_groups, test_groups = hierarchy.group_frame(train, test, "hierarchy")
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=min_frequency,
            sparse_output=True,
        )
        residual_matrix = encoder.fit_transform(fit_groups)
        test_residual_matrix = encoder.transform(test_groups)
        residual_model = Ridge(alpha=alpha).fit(residual_matrix, y - inner_oof)
        correction = residual_model.predict(test_residual_matrix)
        champion = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12654.csv")
        champion_log = np.log1p(champion.SalePrice.to_numpy(dtype=float))
        candidate_price = np.expm1(champion_log + scale * correction)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Support-floor candidate contains invalid prices")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
        pd.DataFrame(
            {
                "Id": test.Id,
                "champion_price": champion.SalePrice,
                "xgb_base_log": full_model.predict(test_matrix),
                "residual_correction_log": correction,
                "ridge_alpha": alpha,
                "minimum_frequency": min_frequency,
                "scale": scale,
                "candidate_price": candidate_price,
            }
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P5-SUPPORT-FLOOR-CHAMPION-TRANSFER",
        "hypothesis": "Higher category pooling/support and ridge shrinkage will retain champion residual gains while preventing low-support market cells from dominating.",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "configurations": len(discovery_scores),
        "selected_on_discovery_only": {"ridge_alpha": alpha, "minimum_frequency": min_frequency, "scale": scale},
        "confirmation_baseline": base_confirmation,
        "confirmation_candidate": selected_confirmation,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": per_seed,
        "confirmation_seeds_improved": seed_wins,
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "components_path": str(components_path.relative_to(ROOT)) if candidate_written else None,
        "external_champion": 0.12654,
        "external_score": None,
        "oof_path": str(output_oof.relative_to(ROOT)),
        "decision": "SHADOW_ONLY" if eligible else "REJECT",
        "caveat": "Validation transfers residual corrections to the champion's fixed seed-42 OOF; the full CatBoost/XGBoost model was not repeated across these seeds.",
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
                "change": "Residual hierarchy champion transfer, min-frequency pooling 10/20, Ridge alpha 10/50",
                "cv": f"{selected_confirmation['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirmation['fold_rmse_std']:.6f}",
                "validation_scheme": "Discovery seeds 2031/2032; confirmation 2033/2034/2035; nested XGB residuals",
                "runtime_seconds": round(time.perf_counter() - started, 2),
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected alpha={alpha:g}, minfreq={min_frequency}, scale={scale:g}; gain={gain:.6f}; champion seed42 OOF transfer only",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()