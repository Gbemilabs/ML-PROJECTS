from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (42, 117, 314)
CONFIRMATION_SEEDS = (2026, 2027, 2028)
REPEATS = 2
SPLITS = 5
FEATURE_SCALES = np.asarray([0.35, 0.55, 1.2, 18.0, 18.0, 1.0, 0.75])
BANDWIDTHS = (0.65, 1.0, 1.5)
SHRINKAGES = (2.0, 5.0, 10.0)
MIXING = (0.5, 1.0)
LOCALITIES = ("neighborhood", "global")
TARGETS = ("log_price", "log_price_per_living_sf")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def feature_matrix(frame: pd.DataFrame, medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    living = pd.to_numeric(frame["GrLivArea"], errors="coerce").to_numpy(dtype=float)
    basement = pd.to_numeric(frame["TotalBsmtSF"], errors="coerce").to_numpy(dtype=float)
    quality = pd.to_numeric(frame["OverallQual"], errors="coerce").to_numpy(dtype=float)
    age = (frame["YrSold"] - frame["YearBuilt"]).to_numpy(dtype=float)
    remodel_age = (frame["YrSold"] - frame["YearRemodAdd"]).to_numpy(dtype=float)
    garage = pd.to_numeric(frame["GarageCars"], errors="coerce").to_numpy(dtype=float)
    bath = (
        frame["FullBath"]
        + 0.5 * frame["HalfBath"]
        + frame["BsmtFullBath"]
        + 0.5 * frame["BsmtHalfBath"]
    ).to_numpy(dtype=float)
    matrix = np.column_stack(
        [np.log1p(living), np.log1p(basement), quality, age, remodel_age, garage, bath]
    )
    if medians is not None:
        missing_rows, missing_cols = np.where(~np.isfinite(matrix))
        matrix[missing_rows, missing_cols] = medians[missing_cols]
    neighborhoods = frame["Neighborhood"].fillna("Missing").astype(str).to_numpy()
    return matrix, neighborhoods


def make_configs() -> list[tuple[str, float, float, float, str]]:
    return [
        (locality, bandwidth, shrinkage, mixing, target)
        for locality in LOCALITIES
        for bandwidth in BANDWIDTHS
        for shrinkage in SHRINKAGES
        for mixing in MIXING
        for target in TARGETS
    ]


def config_name(config: tuple[str, float, float, float, str]) -> str:
    locality, bandwidth, shrinkage, mixing, target = config
    return f"{locality}|bw={bandwidth:g}|k={shrinkage:g}|mix={mixing:g}|target={target}"


def make_predictions(
    train_features: np.ndarray,
    valid_features: np.ndarray,
    train_neighborhood: np.ndarray,
    valid_neighborhood: np.ndarray,
    train_y: np.ndarray,
    valid_base: np.ndarray,
    valid_area_log: np.ndarray,
) -> dict[tuple[str, float, float, float, str], np.ndarray]:
    distance = (valid_features[:, None, :] - train_features[None, :, :]) / FEATURE_SCALES
    distance_squared = np.mean(distance * distance, axis=2)
    same_neighborhood = valid_neighborhood[:, None] == train_neighborhood[None, :]
    target_values = {
        "log_price": train_y,
        "log_price_per_living_sf": train_y - train_features[:, 0],
    }
    predictions: dict[tuple[str, float, float, float, str], np.ndarray] = {}

    for locality in LOCALITIES:
        locality_mask = same_neighborhood if locality == "neighborhood" else np.ones_like(same_neighborhood)
        for bandwidth in BANDWIDTHS:
            weights = np.exp(-0.5 * distance_squared / (bandwidth * bandwidth)) * locality_mask
            weight_sum = weights.sum(axis=1)
            weight_square_sum = np.square(weights).sum(axis=1)
            effective_count = np.divide(
                np.square(weight_sum),
                weight_square_sum,
                out=np.zeros_like(weight_sum),
                where=weight_square_sum > 0,
            )
            for target_name, values in target_values.items():
                local_mean = np.divide(
                    weights @ values,
                    weight_sum,
                    out=valid_base.copy(),
                    where=weight_sum > 0,
                )
                if target_name == "log_price_per_living_sf":
                    local_mean += valid_area_log
                for shrinkage in SHRINKAGES:
                    confidence = effective_count / (effective_count + shrinkage)
                    for mixing in MIXING:
                        weight = np.minimum(1.0, mixing * confidence)
                        predictions[(locality, bandwidth, shrinkage, mixing, target_name)] = (
                            valid_base + weight * (local_mean - valid_base)
                        )
    return predictions


def score_config(
    actual: list[float],
    predicted: list[float],
    fold_scores: list[float],
) -> dict[str, float]:
    errors = np.asarray(predicted) - np.asarray(actual)
    return {
        "pooled_rmse": float(np.mean(errors * errors) ** 0.5),
        "mean_fold_rmse": float(np.mean(fold_scores)),
        "fold_rmse_std": float(np.std(fold_scores)),
        "worst_fold_rmse": float(np.max(fold_scores)),
        "median_fold_rmse": float(np.median(fold_scores)),
    }


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    baseline_oof = pd.read_csv(EXPERIMENTS / "baseline" / "oof_predictions.csv")
    baseline_submission = pd.read_csv(EXPERIMENTS / "canonicalized_baseline_submission.csv")
    if not np.array_equal(baseline_oof["Id"].to_numpy(), train["Id"].to_numpy()):
        raise ValueError("Baseline OOF IDs must align exactly with the training data")
    if not np.array_equal(baseline_submission["Id"].to_numpy(), test["Id"].to_numpy()):
        raise ValueError("Baseline submission IDs must align exactly with the test data")

    y = np.log1p(train["SalePrice"].to_numpy(dtype=float))
    base_oof = np.log1p(baseline_oof["SalePrice"].to_numpy(dtype=float))
    base_test = np.log1p(baseline_submission["SalePrice"].to_numpy(dtype=float))
    train_features, neighborhoods = feature_matrix(train)
    test_features, test_neighborhoods = feature_matrix(test, np.nanmedian(train_features, axis=0))
    train_area_log = train_features[:, 0]
    test_area_log = test_features[:, 0]
    configs = make_configs()
    actual_by_stage = {"discovery": [], "confirmation": []}
    base_by_stage = {"discovery": [], "confirmation": []}
    predictions_by_stage = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    folds_by_stage = {
        stage: {config: [] for config in configs}
        for stage in ("discovery", "confirmation")
    }
    baseline_fold_scores = {"discovery": [], "confirmation": []}
    metadata = []

    for stage, seeds in (
        ("discovery", DISCOVERY_SEEDS),
        ("confirmation", CONFIRMATION_SEEDS),
    ):
        for seed in seeds:
            splitter = RepeatedKFold(
                n_splits=SPLITS,
                n_repeats=REPEATS,
                random_state=seed,
            )
            for split_number, (fit_idx, valid_idx) in enumerate(splitter.split(train), start=1):
                medians = np.nanmedian(train_features[fit_idx], axis=0)
                fit_features = train_features[fit_idx].copy()
                valid_features = train_features[valid_idx].copy()
                for matrix in (fit_features, valid_features):
                    missing_rows, missing_cols = np.where(~np.isfinite(matrix))
                    matrix[missing_rows, missing_cols] = medians[missing_cols]
                fold_predictions = make_predictions(
                    fit_features,
                    valid_features,
                    neighborhoods[fit_idx],
                    neighborhoods[valid_idx],
                    y[fit_idx],
                    base_oof[valid_idx],
                    train_area_log[valid_idx],
                )
                actual = y[valid_idx]
                base = base_oof[valid_idx]
                actual_by_stage[stage].extend(actual.tolist())
                base_by_stage[stage].extend(base.tolist())
                baseline_fold_scores[stage].append(rmse(actual, base))
                repeat_number = (split_number - 1) // SPLITS + 1
                fold_number = (split_number - 1) % SPLITS + 1
                metadata.extend(
                    {
                        "Id": int(train.iloc[row]["Id"]),
                        "stage": stage,
                        "seed": seed,
                        "repeat": repeat_number,
                        "fold": fold_number,
                        "y_log": float(y[row]),
                        "baseline_oof_log": float(base_oof[row]),
                    }
                    for row in valid_idx
                )
                for config, prediction in fold_predictions.items():
                    predictions_by_stage[stage][config].extend(prediction.tolist())
                    folds_by_stage[stage][config].append(rmse(actual, prediction))
                print(f"{stage} seed={seed} split={split_number}/{SPLITS * REPEATS}", flush=True)

    discovery_scores = {}
    for config in configs:
        discovery_scores[config] = score_config(
            actual_by_stage["discovery"],
            predictions_by_stage["discovery"][config],
            folds_by_stage["discovery"][config],
        )
    selected_config = min(discovery_scores, key=lambda config: discovery_scores[config]["pooled_rmse"])
    baseline_scores = {
        stage: {
            "pooled_rmse": rmse(actual_by_stage[stage], base_by_stage[stage]),
            "mean_fold_rmse": float(np.mean(baseline_fold_scores[stage])),
            "fold_rmse_std": float(np.std(baseline_fold_scores[stage])),
            "worst_fold_rmse": float(np.max(baseline_fold_scores[stage])),
        }
        for stage in ("discovery", "confirmation")
    }

    confirmation_score = score_config(
        actual_by_stage["confirmation"],
        predictions_by_stage["confirmation"][selected_config],
        folds_by_stage["confirmation"][selected_config],
    )
    confirmation_baseline = baseline_scores["confirmation"]
    confirmation_gain = confirmation_baseline["pooled_rmse"] - confirmation_score["pooled_rmse"]
    eligible = (
        confirmation_gain >= 0.0003
        and confirmation_score["worst_fold_rmse"]
        <= confirmation_baseline["worst_fold_rmse"] + 0.01
    )

    output_oof = EXPERIMENTS / "comparable_local_oof.csv"
    output_report = EXPERIMENTS / "comparable_local_validation.json"
    shadow_dir = EXPERIMENTS / "shadow_candidates"
    shadow_candidate = shadow_dir / "neighborhood_comparables.csv"
    if output_oof.exists() or output_report.exists() or (eligible and shadow_candidate.exists()):
        raise FileExistsError("Refusing to overwrite prior comparable-property artifacts")

    selected_predictions = (
        predictions_by_stage["discovery"][selected_config]
        + predictions_by_stage["confirmation"][selected_config]
    )
    output_frame = pd.DataFrame(metadata)
    output_frame["selected_comparable_oof_log"] = selected_predictions
    output_frame.to_csv(output_oof, index=False)

    train_features_filled = train_features.copy()
    fit_medians = np.nanmedian(train_features, axis=0)
    missing_rows, missing_cols = np.where(~np.isfinite(train_features_filled))
    train_features_filled[missing_rows, missing_cols] = fit_medians[missing_cols]
    test_features_filled = test_features.copy()
    missing_rows, missing_cols = np.where(~np.isfinite(test_features_filled))
    test_features_filled[missing_rows, missing_cols] = fit_medians[missing_cols]
    test_comparable = make_predictions(
        train_features_filled,
        test_features_filled,
        neighborhoods,
        test_neighborhoods,
        y,
        base_test,
        test_area_log,
    )[selected_config]
    candidate_written = False
    if eligible:
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow = pd.DataFrame(
            {"Id": test["Id"], "SalePrice": np.expm1(test_comparable)}
        )
        if shadow.columns.tolist() != ["Id", "SalePrice"] or not np.isfinite(shadow.SalePrice).all():
            raise ValueError("Comparable shadow submission failed finite/schema validation")
        shadow.to_csv(shadow_candidate, index=False)
        candidate_written = True

    selected_name = config_name(selected_config)
    discovery_baseline = baseline_scores["discovery"]
    discovery_selected = discovery_scores[selected_config]
    report = {
        "experiment_id": "P4-COMPARABLE-LOCALITY",
        "hypothesis": "Continuous, fold-local comparable homes provide local market information beyond broad group means.",
        "target_firewall": "Validation targets are excluded from neighbor target means; no validation row can be its own comparable.",
        "model_base": "Preserved baseline OOF predictions for validation; canonicalized baseline test predictions for inference.",
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "n_splits": SPLITS,
        "n_repeats_per_seed": REPEATS,
        "candidate_config_count": len(configs),
        "selected_config_from_discovery_only": selected_name,
        "discovery_baseline": discovery_baseline,
        "discovery_selected": discovery_selected,
        "confirmation_baseline": confirmation_baseline,
        "confirmation_selected": confirmation_score,
        "confirmation_pooled_gain": confirmation_gain,
        "shadow_eligibility_gate": "confirmation pooled RMSE gain >= 0.0003 and worst fold no more than 0.01 worse",
        "eligible_for_shadow_submission": bool(eligible),
        "shadow_submission_written": candidate_written,
        "shadow_submission_path": str(shadow_candidate.relative_to(ROOT)) if candidate_written else None,
        "external_champion_score": 0.12654,
        "external_champion_local_oof_available": False,
        "external_score": None,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(output_oof.relative_to(ROOT)),
        "decision": "SHADOW_ONLY" if eligible else "REJECT",
        "caveat": "The external champion's exact artifact and OOF predictions are unavailable; this experiment only challenges the historical local baseline.",
    }
    output_report.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in rows):
        raise ValueError("Comparable experiment already exists in research log")
    reason = (
        f"Discovery selected {selected_name}; confirmation pooled RMSE "
        f"{confirmation_baseline['pooled_rmse']:.6f} -> {confirmation_score['pooled_rmse']:.6f}; "
        f"gain={confirmation_gain:.6f}; eligible={eligible}. External champion not locally identifiable."
    )
    row = {
        "experiment_id": report["experiment_id"],
        "hypothesis": report["hypothesis"],
        "change": "Neighborhood-gated continuous comparables with log-price and price-density targets",
        "cv": f"{confirmation_score['pooled_rmse']:.6f}",
        "cv_std": f"{confirmation_score['fold_rmse_std']:.6f}",
        "validation_scheme": "Discovery/confirmation; RepeatedKFold5x2; 3 seeds each",
        "runtime_seconds": report["runtime_seconds"],
        "oof_correlation": "",
        "decision": report["decision"],
        "reason": reason,
        "status": report["decision"],
    }
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(row)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()