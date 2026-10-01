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
DISCOVERY_SEEDS = (19, 43, 89)
CONFIRMATION_SEEDS = (2024, 2025, 2029)
OUTER_SPLITS = 5
INNER_SPLITS = 4
REPRESENTATIONS = ("hierarchy", "hierarchy_plus_size")
RIDGE_ALPHAS = (10.0, 50.0)
MIN_FREQUENCIES = (5, 10)
SCALES = (0.25, 0.5, 0.75, 1.0)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


def summarize(actual: list[float], prediction: list[float], folds: list[float]) -> dict[str, float]:
    errors = np.asarray(prediction) - np.asarray(actual)
    return {
        "pooled_rmse": float(np.mean(errors * errors) ** 0.5),
        "mean_fold_rmse": float(np.mean(folds)),
        "fold_rmse_std": float(np.std(folds)),
        "worst_fold_rmse": float(np.max(folds)),
        "median_fold_rmse": float(np.median(folds)),
    }


def residual_prediction(
    fit_groups: pd.DataFrame,
    valid_groups: pd.DataFrame,
    residual: np.ndarray,
    alpha: float,
    min_frequency: int,
) -> np.ndarray:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=min_frequency,
        sparse_output=True,
    )
    train_matrix = encoder.fit_transform(fit_groups)
    valid_matrix = encoder.transform(valid_groups)
    model = Ridge(alpha=alpha)
    model.fit(train_matrix, residual)
    return model.predict(valid_matrix)


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_seed42.csv")
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12654.csv")
    if not np.array_equal(champion_oof.Id, train.Id):
        raise ValueError("Champion OOF IDs must match the training row order")
    if not np.array_equal(champion_submission.Id, test.Id):
        raise ValueError("Champion submission IDs must match the test row order")

    champion_log = np.log1p(champion_oof.SalePrice.to_numpy(dtype=float))
    champion_test_log = np.log1p(champion_submission.SalePrice.to_numpy(dtype=float))
    X, X_test, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]
    configs = [
        (representation, alpha, min_frequency)
        for representation in REPRESENTATIONS
        for alpha in RIDGE_ALPHAS
        for min_frequency in MIN_FREQUENCIES
    ]
    actual_by_stage = {"discovery": [], "confirmation": []}
    champion_by_stage = {"discovery": [], "confirmation": []}
    metadata: list[pd.DataFrame] = []
    corrections = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    fold_corrections = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }

    for stage, seeds in (("discovery", DISCOVERY_SEEDS), ("confirmation", CONFIRMATION_SEEDS)):
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
                actual = y[outer_valid]
                actual_by_stage[stage].extend(actual.tolist())
                champion_by_stage[stage].extend(champion_log[outer_valid].tolist())
                metadata.append(
                    pd.DataFrame(
                        {
                            "Id": train.iloc[outer_valid].Id.to_numpy(dtype=int),
                            "stage": stage,
                            "seed": seed,
                            "fold": outer_fold,
                            "y_log": actual,
                            "champion_oof_log": champion_log[outer_valid],
                        }
                    )
                )
                group_cache = {
                    representation: hierarchy.group_frame(
                        raw_train,
                        raw_valid,
                        representation,
                    )
                    for representation in REPRESENTATIONS
                }
                for config in configs:
                    representation, alpha, min_frequency = config
                    fit_groups, valid_groups = group_cache[representation]
                    correction = residual_prediction(
                        fit_groups,
                        valid_groups,
                        residual,
                        alpha,
                        min_frequency,
                    )
                    corrections[stage][config].extend(correction.tolist())
                    fold_corrections[stage][config].append(
                        rmse(actual, champion_log[outer_valid] + correction)
                    )
                print(f"{stage} seed={seed} outer fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    discovery_candidates = {}
    confirmation_candidates = {}
    for config in configs:
        for scale in SCALES:
            key = (*config, scale)
            discovery_prediction = np.asarray(champion_by_stage["discovery"]) + scale * np.asarray(
                corrections["discovery"][config]
            )
            confirmation_prediction = np.asarray(champion_by_stage["confirmation"]) + scale * np.asarray(
                corrections["confirmation"][config]
            )
            discovery_fold_scores = []
            confirmation_fold_scores = []
            cursor = 0
            for _seed in DISCOVERY_SEEDS:
                for _fold in range(OUTER_SPLITS):
                    fold_size = len(train) // OUTER_SPLITS
                    if _fold < len(train) % OUTER_SPLITS:
                        fold_size += 1
                    actual_slice = np.asarray(actual_by_stage["discovery"])[cursor : cursor + fold_size]
                    discovery_fold_scores.append(
                        rmse(actual_slice, discovery_prediction[cursor : cursor + fold_size])
                    )
                    cursor += fold_size
            cursor = 0
            for _seed in CONFIRMATION_SEEDS:
                for _fold in range(OUTER_SPLITS):
                    fold_size = len(train) // OUTER_SPLITS
                    if _fold < len(train) % OUTER_SPLITS:
                        fold_size += 1
                    actual_slice = np.asarray(actual_by_stage["confirmation"])[cursor : cursor + fold_size]
                    confirmation_fold_scores.append(
                        rmse(actual_slice, confirmation_prediction[cursor : cursor + fold_size])
                    )
                    cursor += fold_size
            discovery_candidates[key] = summarize(
                actual_by_stage["discovery"],
                discovery_prediction.tolist(),
                discovery_fold_scores,
            )
            confirmation_candidates[key] = summarize(
                actual_by_stage["confirmation"],
                confirmation_prediction.tolist(),
                confirmation_fold_scores,
            )

    selected = min(discovery_candidates, key=lambda key: discovery_candidates[key]["pooled_rmse"])
    selected_confirmation = confirmation_candidates[selected]
    confirmation_base = summarize(
        actual_by_stage["confirmation"],
        champion_by_stage["confirmation"],
        [
            rmse(
                np.asarray(actual_by_stage["confirmation"])[i : i + len(train) // OUTER_SPLITS],
                np.asarray(champion_by_stage["confirmation"])[i : i + len(train) // OUTER_SPLITS],
            )
            for i in range(0, len(actual_by_stage["confirmation"]), len(train) // OUTER_SPLITS)
        ],
    )
    gain = confirmation_base["pooled_rmse"] - selected_confirmation["pooled_rmse"]
    worst_change = selected_confirmation["worst_fold_rmse"] - confirmation_base["worst_fold_rmse"]
    selected_representation, selected_ridge_alpha, selected_min_frequency, selected_scale = selected
    per_seed = {}
    candidate_confirmation = np.asarray(champion_by_stage["confirmation"]) + selected_scale * np.asarray(
        corrections["confirmation"][selected[:3]]
    )
    offset = 0
    for seed in CONFIRMATION_SEEDS:
        seed_slice = slice(offset, offset + len(train))
        seed_actual = np.asarray(actual_by_stage["confirmation"])[seed_slice]
        seed_base = np.asarray(champion_by_stage["confirmation"])[seed_slice]
        seed_candidate = candidate_confirmation[seed_slice]
        per_seed[str(seed)] = {
            "base_rmse": rmse(seed_actual, seed_base),
            "candidate_rmse": rmse(seed_actual, seed_candidate),
        }
        offset += len(train)
    improved_seeds = sum(value["candidate_rmse"] < value["base_rmse"] for value in per_seed.values())
    eligible = gain >= 0.0003 and worst_change <= 0.01 and improved_seeds >= 2

    output_oof = EXPERIMENTS / "phase5_pooled_champion_transfer_oof.csv"
    output_report = EXPERIMENTS / "phase5_pooled_champion_transfer_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase5_pooled_hierarchy_champion.csv"
    components_path = EXPERIMENTS / "phase5_pooled_hierarchy_test_components.csv"
    if output_oof.exists() or output_report.exists() or (
        eligible and (candidate_path.exists() or components_path.exists())
    ):
        raise FileExistsError("Refusing to overwrite pooled hierarchy artifacts")
    selected_correction = np.concatenate(
        [
            np.asarray(corrections["discovery"][selected[:3]]),
            np.asarray(corrections["confirmation"][selected[:3]]),
        ]
    )
    selected_meta = pd.concat(metadata, ignore_index=True)
    selected_meta["residual_correction"] = selected_correction
    selected_meta["candidate_log"] = (
        selected_meta.champion_oof_log + selected_scale * selected_meta.residual_correction
    )
    selected_meta.to_csv(output_oof, index=False)

    candidate_written = False
    if eligible:
        X, X_test, y_series = phase2.prepare_raw(train, test)
        y = y_series.to_numpy(dtype=float)
        model_spec = phase2.model_specs()["XGBoost_regularized"]
        full_prep = phase2.onehot_preprocessor(X)
        train_matrix = full_prep.fit_transform(X)
        test_matrix = full_prep.transform(X_test)
        full_model = clone(model_spec["model"])
        full_model.fit(train_matrix, y)
        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2032)
        for inner_train, inner_valid in inner.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            inner_train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            inner_valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"])
            model.fit(inner_train_matrix, y[inner_train])
            inner_oof[inner_valid] = model.predict(inner_valid_matrix)
        fit_groups, test_groups = hierarchy.group_frame(train, test, selected_representation)
        residual = y - inner_oof
        if selected_representation == "hierarchy":
            test_correction = residual_prediction(
                fit_groups,
                test_groups,
                residual,
                selected_ridge_alpha,
                selected_min_frequency,
            )
        else:
            test_correction = residual_prediction(
                fit_groups,
                test_groups,
                residual,
                selected_ridge_alpha,
                selected_min_frequency,
            )
        champion = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12654.csv")
        champion_log = np.log1p(champion.SalePrice.to_numpy(dtype=float))
        candidate_log = champion_log + selected_scale * test_correction
        candidate_price = np.expm1(candidate_log)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Pooled hierarchy champion candidate has invalid prices")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
        pd.DataFrame(
            {
                "Id": test.Id,
                "champion_price": champion.SalePrice,
                "xgb_base_log": full_model.predict(test_matrix),
                "residual_correction_log": test_correction,
                "scale": selected_scale,
                "candidate_price": candidate_price,
            }
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P5-POOLED-CHAMPION-HIERARCHY",
        "hypothesis": "Pooling sparse categories at the encoder level and regularizing additive group effects should transfer the XGB residual structure to the protected champion without singleton-driven corrections.",
        "champion_oof": "experiments/champion/oof_seed42.csv",
        "outer_residual_firewall": "Residual targets are generated from inner OOF XGBoost predictions in each outer training partition; outer validation labels never fit the residual model.",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "candidate_configs": len(discovery_candidates),
        "selected_config_from_discovery_only": {
            "representation": selected_representation,
            "ridge_alpha": selected_ridge_alpha,
            "minimum_frequency": selected_min_frequency,
            "scale": selected_scale,
        },
        "confirmation_baseline": confirmation_base,
        "confirmation_candidate": selected_confirmation,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": per_seed,
        "confirmation_seeds_improved": improved_seeds,
        "shadow_gate": "Pooled gain >=0.0003, worst fold <=base+0.01, and >=2/3 confirmation seeds improve",
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "components_path": str(components_path.relative_to(ROOT)) if candidate_written else None,
        "external_champion_score": 0.12654,
        "external_score": None,
        "oof_path": str(output_oof.relative_to(ROOT)),
        "decision": "SHADOW_ONLY" if eligible else "REJECT",
        "caveat": "Champion prediction remains one fixed seed-42 OOF crossfit; repeated seeds vary the leakage-safe residual learner rather than retraining the full champion.",
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
                "change": "Ridge hierarchy with rare-level pooling; champion OOF transfer; discovery-only scale selection",
                "cv": f"{selected_confirmation['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirmation['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested residual KFold5 outer/KFold4 inner; 3 discovery and 3 confirmation seeds",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected {selected_representation}, alpha={selected_ridge_alpha:g}, minfreq={selected_min_frequency}, scale={selected_scale:g}; gain={gain:.6f}; fixed champion OOF",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()