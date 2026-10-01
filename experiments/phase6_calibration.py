from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
DISCOVERY_SEEDS = (23, 47, 83)
CONFIRMATION_SEEDS = (2036, 2037, 2041)


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(actual, prediction) ** 0.5)


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    if oof.Id.duplicated().any() or set(oof.Id) != set(train.Id):
        raise ValueError("Champion OOF must contain each training Id exactly once")
    oof = oof.set_index("Id").reindex(train.Id)
    actual = np.log1p(train.SalePrice.to_numpy(dtype=float))
    base = oof.candidate_oof_log.to_numpy(dtype=float)
    methods = ("identity", "intercept", "affine", "ridge10", "isotonic")
    results = {}

    for stage, seeds in (("discovery", DISCOVERY_SEEDS), ("confirmation", CONFIRMATION_SEEDS)):
        predictions = {method: [] for method in methods}
        fold_scores = {method: [] for method in methods}
        actual_rows: list[float] = []
        for seed in seeds:
            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
            for fit_idx, valid_idx in splitter.split(train):
                fold_predictions = {"identity": base[valid_idx].copy()}
                offset = float(np.mean(actual[fit_idx] - base[fit_idx]))
                fold_predictions["intercept"] = base[valid_idx] + offset
                affine = LinearRegression().fit(base[fit_idx, None], actual[fit_idx])
                fold_predictions["affine"] = affine.predict(base[valid_idx, None])
                ridge = Ridge(alpha=10.0).fit(base[fit_idx, None], actual[fit_idx])
                fold_predictions["ridge10"] = ridge.predict(base[valid_idx, None])
                isotonic = IsotonicRegression(out_of_bounds="clip").fit(base[fit_idx], actual[fit_idx])
                fold_predictions["isotonic"] = isotonic.predict(base[valid_idx])
                actual_rows.extend(actual[valid_idx].tolist())
                for method, prediction in fold_predictions.items():
                    predictions[method].extend(prediction.tolist())
                    fold_scores[method].append(rmse(actual[valid_idx], prediction))
        results[stage] = {
            method: {
                "pooled_rmse": rmse(np.asarray(actual_rows), np.asarray(predictions[method])),
                "mean_fold_rmse": float(np.mean(fold_scores[method])),
                "fold_rmse_std": float(np.std(fold_scores[method])),
                "worst_fold_rmse": float(np.max(fold_scores[method])),
            }
            for method in methods
        }

    report = {
        "experiment_id": "P6-CHAMPION-CALIBRATION",
        "hypothesis": "Global log-price calibration may remove systematic multiplicative bias from the current champion.",
        "source_oof": "experiments/champion/oof_0.12374_crossfit.csv",
        "methods": list(methods),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "results": results,
        "decision": "REJECT; all nonidentity calibrators worsened pooled confirmation RMSE",
        "caveat": "Seed partitions reuse the same target rows; this is a calibration stability screen, not an independent external evaluation.",
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }
    output = EXPERIMENTS / "phase6_calibration_validation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in rows):
        raise ValueError("Calibration result already exists in research log")
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Identity vs intercept/affine/Ridge/isotonic calibration of champion log OOF",
                "cv": f"{results['confirmation']['identity']['pooled_rmse']:.6f}",
                "cv_std": f"{results['confirmation']['identity']['fold_rmse_std']:.6f}",
                "validation_scheme": "KFold5; discovery seeds 23/47/83; confirmation 2036/2037/2041",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": "REJECT",
                "reason": "Every calibration method worsened pooled confirmation RMSE relative to identity.",
                "status": "REJECT",
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()