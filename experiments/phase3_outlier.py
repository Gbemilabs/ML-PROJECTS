from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import phase2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
FLAGGED_IDS = (524, 1299)
TEST_ANALOGUE_ID = 2550


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def fit_predict(spec: dict, X_fit: pd.DataFrame, y_fit: pd.Series, X_valid: pd.DataFrame) -> np.ndarray:
    preprocessor = phase2.onehot_preprocessor(X_fit)
    train_matrix = preprocessor.fit_transform(X_fit)
    valid_matrix = preprocessor.transform(X_valid)
    model = clone(spec["model"])
    model.fit(train_matrix, y_fit)
    return model.predict(valid_matrix)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    X, X_test, y = phase2.prepare_raw(train, test)
    flagged = train["Id"].isin(FLAGGED_IDS).to_numpy()
    if int(flagged.sum()) != len(FLAGGED_IDS):
        raise ValueError("Expected both flagged training rows to be present")

    spec = phase2.model_specs()["XGBoost_regularized"]
    baseline_oof = np.zeros(len(X), dtype=float)
    exclude_oof = np.zeros(len(X), dtype=float)
    fold_scores = {"baseline": [], "exclude_flagged": []}
    folds = KFold(n_splits=5, shuffle=True, random_state=42).split(X)

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_valid = X.iloc[valid_idx]
        for name, source_idx, predictions in (
            ("baseline", train_idx, baseline_oof),
            ("exclude_flagged", train_idx[~flagged[train_idx]], exclude_oof),
        ):
            prediction = fit_predict(
                spec,
                X.iloc[source_idx],
                y.iloc[source_idx],
                X_valid,
            )
            predictions[valid_idx] = prediction
            fold_scores[name].append(rmse(y.iloc[valid_idx].to_numpy(), prediction))
        print(f"Completed fold {fold}/5", flush=True)

    flagged_error = {}
    for name, prediction in (("baseline", baseline_oof), ("exclude_flagged", exclude_oof)):
        error = prediction - y.to_numpy()
        flagged_error[name] = {
            "pooled_rmse": rmse(y.to_numpy(), prediction),
            "nonflagged_pooled_rmse": rmse(y.to_numpy()[~flagged], prediction[~flagged]),
            "flagged_rmse": rmse(y.to_numpy()[flagged], prediction[flagged]),
            "fold_rmse": fold_scores[name],
        }

    test_row = np.flatnonzero(test["Id"].to_numpy() == TEST_ANALOGUE_ID)
    if len(test_row) != 1:
        raise ValueError("Expected exactly one matching test analogue")
    test_prices = {}
    for name, indices in (
        ("baseline", np.arange(len(X))),
        ("exclude_flagged", np.flatnonzero(~flagged)),
    ):
        prediction = fit_predict(spec, X.iloc[indices], y.iloc[indices], X_test)
        test_prices[name] = float(np.expm1(prediction[test_row[0]]))

    output_oof = EXPERIMENTS / "outlier_exclusion_oof.csv"
    output_report = EXPERIMENTS / "outlier_exclusion_validation.json"
    if output_oof.exists() or output_report.exists():
        raise FileExistsError("Refusing to overwrite existing outlier experiment artifacts")

    oof_frame = pd.DataFrame(
        {
            "Id": train["Id"],
            "y_log": y,
            "baseline_oof_log": baseline_oof,
            "exclude_524_1299_oof_log": exclude_oof,
            "is_flagged": flagged,
        }
    )
    report = {
        "experiment_id": "P3-OUTLIER-EXCLUSION",
        "model": "XGBoost_regularized",
        "validation": "KFold5 shuffled seed42; paired training-fold exclusion; flagged rows retained in validation",
        "flagged_ids": list(FLAGGED_IDS),
        "test_analogue_id": TEST_ANALOGUE_ID,
        "scores": flagged_error,
        "test_analogue_price_prediction": test_prices,
        "oof_path": str(output_oof.relative_to(ROOT)),
        "decision": "Do not globally delete: OOF gain conflicts with close test-archetype risk.",
    }
    oof_frame.to_csv(output_oof, index=False)
    output_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()