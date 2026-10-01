from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SEEDS = (42, 117, 314, 999, 2026)
N_SPLITS = 5
N_REPEATS = 2
ALPHA = 1.0


def archetype_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["Neighborhood"].eq("Edwards")
        & frame["OverallQual"].eq(10)
        & frame["YearBuilt"].ge(2007)
        & frame["SaleType"].eq("New")
        & frame["SaleCondition"].eq("Partial")
    ).to_numpy()


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    baseline_oof = pd.read_csv(ROOT / "experiments" / "baseline" / "oof_predictions.csv")
    candidate = pd.read_csv(EXPERIMENTS / "canonicalized_baseline_submission.csv")

    oof_price = baseline_oof.set_index("Id")["SalePrice"].reindex(train["Id"])
    if oof_price.isna().any() or not np.array_equal(baseline_oof["Id"].to_numpy(), train["Id"].to_numpy()):
        raise ValueError("Baseline OOF rows must align exactly with the training file")
    if not np.array_equal(candidate["Id"].to_numpy(), test["Id"].to_numpy()):
        raise ValueError("Candidate submission IDs must align exactly with the test file")

    y_log = np.log1p(train["SalePrice"].to_numpy(dtype=float))
    base_oof_log = np.log1p(oof_price.to_numpy(dtype=float))
    train_group = archetype_mask(train)
    test_group = archetype_mask(test)
    if not train_group.any() or not test_group.any():
        raise ValueError("The Edwards new-construction archetype is absent from train or test")

    prediction_sum = np.zeros(len(train), dtype=float)
    prediction_count = np.zeros(len(train), dtype=int)
    peer_count = np.zeros(len(train), dtype=int)
    repeated_actual: list[float] = []
    repeated_base: list[float] = []
    repeated_corrected: list[float] = []
    repeated_group_actual: list[float] = []
    repeated_group_base: list[float] = []
    repeated_group_corrected: list[float] = []
    fold_scores_base: list[float] = []
    fold_scores_corrected: list[float] = []
    seed_reports = []

    # The two archetype rows were held out together in the saved baseline OOF split.
    # Repeated folds therefore test only the fold-local target correction, not a refit.
    for seed in SEEDS:
        seed_base_scores = []
        seed_corrected_scores = []
        seed_peer_rows = 0
        splits = RepeatedKFold(
            n_splits=N_SPLITS,
            n_repeats=N_REPEATS,
            random_state=seed,
        ).split(train)
        for fit_idx, valid_idx in splits:
            valid_prediction = base_oof_log[valid_idx].copy()
            corrected = valid_prediction.copy()
            valid_group = train_group[valid_idx]
            fit_group = train_group[fit_idx]

            if valid_group.any() and fit_group.any():
                group_mean = float(y_log[fit_idx[fit_group]].mean())
                corrected[valid_group] += ALPHA * (
                    group_mean - corrected[valid_group]
                )
                peer_count[valid_idx[valid_group]] += 1
                seed_peer_rows += int(valid_group.sum())

            actual = y_log[valid_idx]
            base_score = rmse(actual, valid_prediction)
            corrected_score = rmse(actual, corrected)
            seed_base_scores.append(base_score)
            seed_corrected_scores.append(corrected_score)
            fold_scores_base.append(base_score)
            fold_scores_corrected.append(corrected_score)

            prediction_sum[valid_idx] += corrected
            prediction_count[valid_idx] += 1
            repeated_actual.extend(actual.tolist())
            repeated_base.extend(valid_prediction.tolist())
            repeated_corrected.extend(corrected.tolist())
            if valid_group.any():
                repeated_group_actual.extend(actual[valid_group].tolist())
                repeated_group_base.extend(valid_prediction[valid_group].tolist())
                repeated_group_corrected.extend(corrected[valid_group].tolist())

        seed_reports.append(
            {
                "seed": seed,
                "mean_fold_rmse_baseline_oof": float(np.mean(seed_base_scores)),
                "mean_fold_rmse_corrected": float(np.mean(seed_corrected_scores)),
                "peer_supported_group_rows": seed_peer_rows,
            }
        )

    repeated_actual_array = np.asarray(repeated_actual)
    repeated_base_array = np.asarray(repeated_base)
    repeated_corrected_array = np.asarray(repeated_corrected)
    repeated_group_actual_array = np.asarray(repeated_group_actual)
    repeated_group_base_array = np.asarray(repeated_group_base)
    repeated_group_corrected_array = np.asarray(repeated_group_corrected)
    row_mean_oof = prediction_sum / prediction_count

    output_oof = EXPERIMENTS / "edwards_archetype_oof.csv"
    output_candidate = EXPERIMENTS / "edwards_archetype_candidate_submission.csv"
    output_report = EXPERIMENTS / "edwards_archetype_validation.json"
    if any(path.exists() for path in (output_oof, output_candidate, output_report)):
        raise FileExistsError("Refusing to overwrite an existing archetype experiment artifact")

    oof_frame = pd.DataFrame(
        {
            "Id": train["Id"],
            "SalePrice": train["SalePrice"],
            "y_log": y_log,
            "baseline_oof_log": base_oof_log,
            "repeated_mean_corrected_oof_log": row_mean_oof,
            "is_archetype": train_group,
            "peer_supported_validation_count": peer_count,
            "validation_prediction_count": prediction_count,
        }
    )

    candidate_log = np.log1p(candidate["SalePrice"].to_numpy(dtype=float))
    target_group_mean = float(y_log[train_group].mean())
    candidate_log[test_group] += ALPHA * (
        target_group_mean - candidate_log[test_group]
    )
    candidate_prices = np.expm1(candidate_log)
    if not np.isfinite(candidate_prices).all() or (candidate_prices <= 0).any():
        raise ValueError("Corrected candidate contains invalid SalePrice predictions")

    candidate_frame = pd.DataFrame(
        {"Id": test["Id"], "SalePrice": candidate_prices}
    )
    if candidate_frame.columns.tolist() != ["Id", "SalePrice"]:
        raise ValueError("Candidate submission columns are invalid")

    report = {
        "experiment_id": "P3-EDWARDS-ARCHETYPE-TE",
        "hypothesis": "A fold-trained neighborhood/quality/new-sale archetype mean can correct a rare test-matched regime missed by the global model.",
        "target_encoding": "mean log SalePrice from training-fold rows matching Edwards, OverallQual=10, YearBuilt>=2007, SaleType=New, SaleCondition=Partial",
        "alpha": ALPHA,
        "model_base": "Preserved baseline OOF predictions; target correction evaluated on repeated folds only",
        "unique_train_group_rows": int(train_group.sum()),
        "test_group_rows": int(test_group.sum()),
        "group_mean_saleprice_geometric": float(np.expm1(target_group_mean)),
        "seeds": list(SEEDS),
        "n_splits": N_SPLITS,
        "n_repeats_per_seed": N_REPEATS,
        "peer_supported_validation_rows": int(peer_count.sum()),
        "validation_rows_with_group_target_peer": int(len(repeated_group_actual)),
        "repeated_pooled_rmse_baseline_oof": rmse(
            repeated_actual_array, repeated_base_array
        ),
        "repeated_pooled_rmse_corrected": rmse(
            repeated_actual_array, repeated_corrected_array
        ),
        "repeated_group_rmse_baseline_oof": rmse(
            repeated_group_actual_array, repeated_group_base_array
        ),
        "repeated_group_rmse_corrected": rmse(
            repeated_group_actual_array, repeated_group_corrected_array
        ),
        "row_mean_oof_rmse_corrected": rmse(y_log, row_mean_oof),
        "mean_fold_rmse_baseline_oof": float(np.mean(fold_scores_base)),
        "mean_fold_rmse_corrected": float(np.mean(fold_scores_corrected)),
        "seed_reports": seed_reports,
        "candidate_adjusted_ids": test.loc[test_group, "Id"].astype(int).tolist(),
        "candidate_adjusted_saleprice": candidate_frame.loc[test_group, "SalePrice"].tolist(),
        "oof_path": str(output_oof.relative_to(ROOT)),
        "candidate_path": str(output_candidate.relative_to(ROOT)),
        "external_score": None,
        "decision": "INVESTIGATE; only two unique training examples and no external score",
    }

    oof_frame.to_csv(output_oof, index=False)
    candidate_frame.to_csv(output_candidate, index=False)
    output_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()