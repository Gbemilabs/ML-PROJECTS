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
SEEDS = (42, 117, 314, 2026, 2027)
DEVELOPMENT_SEEDS = (42, 117, 314)
CONFIRMATION_SEEDS = (2026, 2027)
OUTER_SPLITS = 5
INNER_SPLITS = 4
TARGETS = ("direct", "GrLivArea", "TotalSF", "UsableArea")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def log_area(frame: pd.DataFrame, target_name: str) -> np.ndarray:
    living = frame["GrLivArea"].to_numpy(dtype=float)
    basement = frame["TotalBsmtSF"].to_numpy(dtype=float)
    if target_name == "direct":
        return np.zeros(len(frame), dtype=float)
    if target_name == "GrLivArea":
        area = living
    elif target_name == "TotalSF":
        area = frame["1stFlrSF"].to_numpy(dtype=float) + frame["2ndFlrSF"].to_numpy(dtype=float) + basement
    elif target_name == "UsableArea":
        area = living + 0.5 * basement
    else:
        raise ValueError(f"Unknown target geometry: {target_name}")
    return np.log(np.maximum(area, 1.0))


def fit_target_model(
    spec: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_offset: np.ndarray,
    X_valid: pd.DataFrame,
    valid_offset: np.ndarray,
) -> np.ndarray:
    preprocessor = phase2.onehot_preprocessor(X_train)
    train_matrix = preprocessor.fit_transform(X_train)
    valid_matrix = preprocessor.transform(X_valid)
    model = clone(spec["model"])
    model.fit(train_matrix, y_train - train_offset)
    return model.predict(valid_matrix) + valid_offset


def metric_summary(actual: np.ndarray, predicted: np.ndarray, fold_scores: list[float]) -> dict:
    return {
        "pooled_rmse": rmse(actual, predicted),
        "mean_fold_rmse": float(np.mean(fold_scores)),
        "fold_rmse_std": float(np.std(fold_scores)),
        "worst_fold_rmse": float(np.max(fold_scores)),
        "median_fold_rmse": float(np.median(fold_scores)),
    }


def main() -> None:
    started = time.perf_counter()
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    X, _, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    spec = phase2.model_specs()["XGBoost_regularized"]
    historical_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_seed42.csv")
    if not np.array_equal(historical_oof.Id.to_numpy(), train.Id.to_numpy()):
        raise ValueError("Champion OOF IDs do not align with training rows")
    champion_log = np.log1p(historical_oof.SalePrice.to_numpy(dtype=float))

    records = []
    fold_records = []
    selected_targets: dict[str, int] = {target: 0 for target in TARGETS}
    for seed in SEEDS:
        outer = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=seed)
        for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer.split(X), start=1):
            X_outer_train = X.iloc[outer_train_idx]
            X_outer_valid = X.iloc[outer_valid_idx]
            y_outer_train = y[outer_train_idx]
            y_outer_valid = y[outer_valid_idx]
            inner_oof = {target: np.zeros(len(outer_train_idx), dtype=float) for target in TARGETS}
            inner = KFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=seed * 100 + outer_fold,
            )

            for inner_train_idx, inner_valid_idx in inner.split(X_outer_train):
                X_inner_train = X_outer_train.iloc[inner_train_idx]
                X_inner_valid = X_outer_train.iloc[inner_valid_idx]
                for target_name in TARGETS:
                    train_offset = log_area(train.iloc[outer_train_idx].iloc[inner_train_idx], target_name)
                    valid_offset = log_area(train.iloc[outer_train_idx].iloc[inner_valid_idx], target_name)
                    inner_oof[target_name][inner_valid_idx] = fit_target_model(
                        spec,
                        X_inner_train,
                        y_outer_train[inner_train_idx],
                        train_offset,
                        X_inner_valid,
                        valid_offset,
                    )

            inner_scores = {
                target_name: rmse(y_outer_train, prediction)
                for target_name, prediction in inner_oof.items()
            }
            selected_target = min(inner_scores, key=inner_scores.get)
            selected_targets[selected_target] += 1

            outer_train_offset = log_area(train.iloc[outer_train_idx], selected_target)
            outer_valid_offset = log_area(train.iloc[outer_valid_idx], selected_target)
            selected_prediction = fit_target_model(
                spec,
                X_outer_train,
                y_outer_train,
                outer_train_offset,
                X_outer_valid,
                outer_valid_offset,
            )
            if selected_target == "direct":
                direct_prediction = selected_prediction
            else:
                direct_prediction = fit_target_model(
                    spec,
                    X_outer_train,
                    y_outer_train,
                    np.zeros(len(outer_train_idx)),
                    X_outer_valid,
                    np.zeros(len(outer_valid_idx)),
                )

            selected_score = rmse(y_outer_valid, selected_prediction)
            direct_score = rmse(y_outer_valid, direct_prediction)
            fold_records.append(
                {
                    "seed": seed,
                    "fold": outer_fold,
                    "selected_target": selected_target,
                    "selected_rmse": selected_score,
                    "direct_rmse": direct_score,
                    "inner_scores": inner_scores,
                }
            )
            for offset, row_index in enumerate(outer_valid_idx):
                records.append(
                    {
                        "Id": int(train.iloc[row_index].Id),
                        "seed": seed,
                        "stage": "development" if seed in DEVELOPMENT_SEEDS else "confirmation",
                        "fold": outer_fold,
                        "selected_target": selected_target,
                        "y_log": float(y_outer_valid[offset]),
                        "champion_oof_log": float(champion_log[row_index]),
                        "direct_xgb_log": float(direct_prediction[offset]),
                        "selected_price_density_log": float(selected_prediction[offset]),
                    }
                )
            print(
                f"seed={seed} outer fold={outer_fold}/{OUTER_SPLITS} selected={selected_target}",
                flush=True,
            )

    result = pd.DataFrame(records)
    oof_path = EXPERIMENTS / "price_density_nested_oof.csv"
    report_path = EXPERIMENTS / "price_density_validation.json"
    if oof_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite prior price-density artifacts")
    result.to_csv(oof_path, index=False)

    def summarize(stage: str, column: str) -> dict:
        selected = result.loc[result.stage.eq(stage)]
        fold_scores = selected.groupby(["seed", "fold"], sort=False).apply(
            lambda fold: rmse(fold.y_log.to_numpy(), fold[column].to_numpy()),
            include_groups=False,
        ).to_numpy()
        return metric_summary(selected.y_log.to_numpy(), selected[column].to_numpy(), fold_scores.tolist())

    stage_metrics = {
        stage: {
            "historical_champion_oof": summarize(stage, "champion_oof_log"),
            "direct_xgb": summarize(stage, "direct_xgb_log"),
            "nested_price_density_selection": summarize(stage, "selected_price_density_log"),
        }
        for stage in ("development", "confirmation")
    }
    candidate_eligible = (
        stage_metrics["confirmation"]["nested_price_density_selection"]["pooled_rmse"]
        <= stage_metrics["confirmation"]["direct_xgb"]["pooled_rmse"] - 0.0003
        and stage_metrics["confirmation"]["nested_price_density_selection"]["worst_fold_rmse"]
        <= stage_metrics["confirmation"]["direct_xgb"]["worst_fold_rmse"] + 0.01
        and stage_metrics["confirmation"]["nested_price_density_selection"]["pooled_rmse"]
        < stage_metrics["confirmation"]["historical_champion_oof"]["pooled_rmse"]
    )

    report = {
        "experiment_id": "P4-PRICE-DENSITY-NESTED",
        "hypothesis": "Log price can be decomposed into a property-specific area term plus learnable price density.",
        "target_firewall": "Area target form is selected from inner OOF predictions inside each outer training fold; outer validation labels are never used for selection.",
        "base_model": "XGBoost_regularized, fixed phase-two parameters",
        "target_candidates": list(TARGETS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "selection_counts_by_outer_fold": selected_targets,
        "fold_records": fold_records,
        "metrics": stage_metrics,
        "shadow_candidate_gate": "On confirmation: nested-selected target beats direct XGBoost by >=0.0003 pooled, worst fold no >0.01 worse, and pooled RMSE is below champion seed-42 OOF.",
        "eligible_for_shadow_submission": bool(candidate_eligible),
        "external_champion_score": 0.12654,
        "external_score": None,
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "decision": "SHADOW_ELIGIBLE_ONLY" if candidate_eligible else "REJECT",
    }
    report_path.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    if any(row.get("experiment_id") == report["experiment_id"] for row in log_rows):
        raise ValueError("Price-density experiment already exists in research log")
    confirm = stage_metrics["confirmation"]["nested_price_density_selection"]
    base_confirm = stage_metrics["confirmation"]["direct_xgb"]
    with log_path.open("a", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writerow(
            {
                "experiment_id": report["experiment_id"],
                "hypothesis": report["hypothesis"],
                "change": "Nested selection among log-price, log(price/area), three structural area definitions",
                "cv": f"{confirm['pooled_rmse']:.6f}",
                "cv_std": f"{confirm['fold_rmse_std']:.6f}",
                "validation_scheme": "Nested KFold5 outer/KFold4 inner; 3 development and 2 confirmation seeds",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Confirmation direct={base_confirm['pooled_rmse']:.6f}, nested area target={confirm['pooled_rmse']:.6f}; compared with champion OOF {stage_metrics['confirmation']['historical_champion_oof']['pooled_rmse']:.6f}",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()