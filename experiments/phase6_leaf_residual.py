from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import phase2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (29, 59, 97)
CONFIRMATION_SEEDS = (2038, 2039, 2040)
OUTER_SPLITS = 5
INNER_SPLITS = 4
TOP_K = (5, 15, 30)
POWERS = (1.0, 2.0)
SHRINKAGES = (2.0, 5.0, 10.0)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


def leaf_similarity(train_leaves: np.ndarray, query_leaves: np.ndarray) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    train_matrix = encoder.fit_transform(train_leaves)
    query_matrix = encoder.transform(query_leaves)
    return (query_matrix @ train_matrix.T).toarray() / train_leaves.shape[1]


def local_residual_prediction(
    similarity: np.ndarray,
    residual: np.ndarray,
    top_k: int,
    power: float,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.argpartition(similarity, -top_k, axis=1)[:, -top_k:]
    selected_similarity = np.take_along_axis(similarity, selected, axis=1)
    weights = np.power(np.maximum(selected_similarity, 0.0), power)
    weight_sum = weights.sum(axis=1)
    weight_square_sum = np.square(weights).sum(axis=1)
    local_mean = np.divide(
        np.sum(weights * residual[selected], axis=1),
        weight_sum,
        out=np.zeros(len(similarity), dtype=float),
        where=weight_sum > 0,
    )
    effective_count = np.divide(
        weight_sum * weight_sum,
        weight_square_sum,
        out=np.zeros_like(weight_sum),
        where=weight_square_sum > 0,
    )
    correction = effective_count / (effective_count + shrinkage) * local_mean
    return correction, effective_count


def summarize(actual: list[float], prediction: list[float], folds: list[float]) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(actual)
    return {
        "pooled_rmse": float(np.mean(error * error) ** 0.5),
        "mean_fold_rmse": float(np.mean(folds)),
        "fold_rmse_std": float(np.std(folds)),
        "worst_fold_rmse": float(np.max(folds)),
        "median_fold_rmse": float(np.median(folds)),
    }


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
    if champion_oof.Id.duplicated().any() or set(champion_oof.Id) != set(train.Id):
        raise ValueError("Champion OOF must contain all training IDs exactly once")
    champion_oof = champion_oof.set_index("Id").reindex(train.Id).reset_index()
    if not np.array_equal(champion_submission.Id, test.Id):
        raise ValueError("Champion submission IDs do not align with test IDs")

    X, X_test, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    base_log = champion_oof.candidate_oof_log.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]
    configs = [(top_k, power, shrinkage) for top_k in TOP_K for power in POWERS for shrinkage in SHRINKAGES]
    stage_data = {"discovery": {config: [] for config in configs}, "confirmation": {config: [] for config in configs}}
    actual_by_stage = {"discovery": [], "confirmation": []}
    base_by_stage = {"discovery": [], "confirmation": []}
    fold_scores = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
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
                    inner_model = clone(model_spec["model"])
                    inner_model.fit(train_matrix, y_train[inner_train])
                    inner_oof[inner_valid] = inner_model.predict(valid_matrix)
                residual = y_train - inner_oof

                outer_preprocessor = phase2.onehot_preprocessor(X_train)
                outer_train_matrix = outer_preprocessor.fit_transform(X_train)
                outer_valid_matrix = outer_preprocessor.transform(X_valid)
                outer_model = clone(model_spec["model"])
                outer_model.fit(outer_train_matrix, y_train)
                train_leaves = outer_model.apply(outer_train_matrix)
                valid_leaves = outer_model.apply(outer_valid_matrix)
                similarity = leaf_similarity(train_leaves, valid_leaves)
                actual = y[outer_valid]
                base_prediction = base_log[outer_valid]
                actual_by_stage[stage].extend(actual.tolist())
                base_by_stage[stage].extend(base_prediction.tolist())
                metadata.append(
                    pd.DataFrame(
                        {
                            "Id": train.iloc[outer_valid].Id.to_numpy(dtype=int),
                            "stage": stage,
                            "seed": seed,
                            "fold": outer_fold,
                            "y_log": actual,
                            "champion_oof_log": base_prediction,
                        }
                    )
                )
                for config in configs:
                    correction, effective = local_residual_prediction(
                        similarity,
                        residual,
                        top_k=config[0],
                        power=config[1],
                        shrinkage=config[2],
                    )
                    prediction = base_prediction + correction
                    stage_data[stage][config].extend(correction.tolist())
                    fold_scores[stage][config].append(rmse(actual, prediction))
                print(f"{stage} seed={seed} fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    discovery_scores = {}
    confirmation_scores = {}
    for config in configs:
        discovery_prediction = np.asarray(base_by_stage["discovery"]) + np.asarray(stage_data["discovery"][config])
        confirmation_prediction = np.asarray(base_by_stage["confirmation"]) + np.asarray(stage_data["confirmation"][config])
        discovery_scores[config] = summarize(
            actual_by_stage["discovery"],
            discovery_prediction.tolist(),
            fold_scores["discovery"][config],
        )
        confirmation_scores[config] = summarize(
            actual_by_stage["confirmation"],
            confirmation_prediction.tolist(),
            fold_scores["confirmation"][config],
        )
    selected = min(configs, key=lambda config: discovery_scores[config]["pooled_rmse"])
    baseline_fold_scores = []
    for stage in ("discovery", "confirmation"):
        cursor = 0
        n_splits = len(DISCOVERY_SEEDS if stage == "discovery" else CONFIRMATION_SEEDS) * OUTER_SPLITS
        actual = actual_by_stage[stage]
        base_predictions = base_by_stage[stage]
        fold_size = len(train) // OUTER_SPLITS
        for _ in range(n_splits):
            baseline_fold_scores.append(
                rmse(
                    np.asarray(actual[cursor : cursor + fold_size]),
                    np.asarray(base_predictions[cursor : cursor + fold_size]),
                )
            )
            cursor += fold_size
    confirmation_base_scores = []
    cursor = 0
    for _ in range(len(CONFIRMATION_SEEDS) * OUTER_SPLITS):
        n = len(train) // OUTER_SPLITS
        confirmation_base_scores.append(
            rmse(
                np.asarray(actual_by_stage["confirmation"][cursor : cursor + n]),
                np.asarray(base_by_stage["confirmation"][cursor : cursor + n]),
            )
        )
        cursor += n
    baseline_confirmation = summarize(
        actual_by_stage["confirmation"],
        base_by_stage["confirmation"],
        confirmation_base_scores,
    )
    selected_confirmation = confirmation_scores[selected]
    gain = baseline_confirmation["pooled_rmse"] - selected_confirmation["pooled_rmse"]
    worst_change = selected_confirmation["worst_fold_rmse"] - baseline_confirmation["worst_fold_rmse"]
    per_seed = {}
    selected_correction = np.asarray(stage_data["confirmation"][selected])
    offset = 0
    for seed in CONFIRMATION_SEEDS:
        n = len(train)
        actual = np.asarray(actual_by_stage["confirmation"][offset : offset + n])
        base = np.asarray(base_by_stage["confirmation"][offset : offset + n])
        corr = selected_correction[offset : offset + n]
        per_seed[str(seed)] = {"base_rmse": rmse(actual, base), "candidate_rmse": rmse(actual, base + corr)}
        offset += n
    seed_wins = sum(item["candidate_rmse"] < item["base_rmse"] for item in per_seed.values())
    eligible = gain >= 0.0003 and worst_change <= 0.01 and seed_wins >= 2

    out_oof = EXPERIMENTS / "phase6_leaf_residual_oof.csv"
    out_report = EXPERIMENTS / "phase6_leaf_residual_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase6_leaf_residual.csv"
    components_path = EXPERIMENTS / "phase6_leaf_residual_test_components.csv"
    if out_oof.exists() or out_report.exists() or (eligible and (candidate_path.exists() or components_path.exists())):
        raise FileExistsError("Refusing to overwrite phase-six leaf-similarity artifacts")
    combined_oof = pd.concat(metadata, ignore_index=True)
    combined_oof["residual_correction"] = np.concatenate(
        [stage_data["discovery"][selected], stage_data["confirmation"][selected]]
    )
    combined_oof["candidate_log"] = combined_oof.champion_oof_log + combined_oof.residual_correction
    combined_oof.to_csv(out_oof, index=False)

    candidate_written = False
    if eligible:
        full_prep = phase2.onehot_preprocessor(X)
        full_train_matrix = full_prep.fit_transform(X)
        test_matrix = full_prep.transform(X_test)
        full_model = clone(model_spec["model"]).fit(full_train_matrix, y)
        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2041)
        for inner_train, inner_valid in inner.split(X):
            prep = phase2.onehot_preprocessor(X.iloc[inner_train])
            train_matrix = prep.fit_transform(X.iloc[inner_train])
            valid_matrix = prep.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"]).fit(train_matrix, y[inner_train])
            inner_oof[inner_valid] = model.predict(valid_matrix)
        full_residual = y - inner_oof
        train_leaves = full_model.apply(full_train_matrix)
        test_leaves = full_model.apply(test_matrix)
        test_similarity = leaf_similarity(train_leaves, test_leaves)
        correction, effective = local_residual_prediction(
            test_similarity,
            full_residual,
            top_k=selected[0],
            power=selected[1],
            shrinkage=selected[2],
        )
        champion = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")
        candidate_price = np.expm1(np.log1p(champion.SalePrice.to_numpy(dtype=float)) + correction)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Leaf-similarity candidate has invalid prices")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
        pd.DataFrame(
            {"Id": test.Id, "champion_price": champion.SalePrice, "correction_log": correction, "effective_neighbor_count": effective, "candidate_price": candidate_price}
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P6-TREE-LEAF-RESIDUAL-NEIGHBORS",
        "hypothesis": "Properties sharing terminal leaves across a fitted tree ensemble have residual similarity beyond raw feature distance.",
        "leaf_embedding": "outer-fold XGBoost terminal leaf IDs; validation labels are excluded from the fit",
        "residual_target": "outer-training y_log minus inner-OOF XGBoost prediction",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "top_k": list(TOP_K),
        "powers": list(POWERS),
        "shrinkages": list(SHRINKAGES),
        "candidate_configs": len(configs),
        "selected_on_discovery_only": {"top_k": selected[0], "power": selected[1], "shrinkage": selected[2]},
        "confirmation_baseline": baseline_confirmation,
        "confirmation_candidate": selected_confirmation,
        "confirmation_gain": gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": per_seed,
        "confirmation_seeds_improved": seed_wins,
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "external_champion": 0.12374,
        "external_score": None,
        "decision": "SHADOW_ONLY" if eligible else "REJECT",
        "caveat": "Residual labels are inner-OOF, while leaf features are derived from the outer model; this is honest on outer validation but can still overfit training leaf partitions. Treat as candidate evidence only.",
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
                "change": "Tree-leaf similarity weighted inner-OOF residual correction",
                "cv": f"{selected_confirmation['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirmation['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested KFold5 outer/KFold4 inner; discovery/confirmation 3 seeds each",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected topk={selected[0]}, power={selected[1]:g}, shrinkage={selected[2]:g}; gain={gain:.6f}; champion OOF transfer",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()