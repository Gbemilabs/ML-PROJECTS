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
RIDGE_ALPHA = 10.0
MIN_FREQUENCY = 2
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("all_residuals", "winsorize_2_98")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def fit_residual_correction(
    groups_train: pd.DataFrame,
    groups_valid: pd.DataFrame,
    residual: np.ndarray,
    variant: str,
) -> np.ndarray:
    if variant == "winsorize_2_98":
        lower, upper = np.quantile(residual, [0.02, 0.98])
        residual = np.clip(residual, lower, upper)
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=MIN_FREQUENCY,
        sparse_output=True,
    )
    train_matrix = encoder.fit_transform(groups_train)
    valid_matrix = encoder.transform(groups_valid)
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(train_matrix, residual)
    return model.predict(valid_matrix)


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
    if not np.array_equal(champion_oof.Id.to_numpy(), train.Id.to_numpy()):
        raise ValueError("Champion OOF IDs must align exactly with training rows")
    champion_log_by_id = dict(
        zip(champion_oof.Id, np.log1p(champion_oof.SalePrice.to_numpy(dtype=float)))
    )
    target_log_by_id = dict(zip(train.Id, np.log1p(train.SalePrice.to_numpy(dtype=float))))
    X, X_test, y_series = phase2.prepare_raw(train, test)
    y = y_series.to_numpy(dtype=float)
    model_spec = phase2.model_specs()["XGBoost_regularized"]

    records = []
    for seed in DISCOVERY_SEEDS:
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
            train_groups, valid_groups = hierarchy.group_frame(
                raw_train,
                raw_valid,
                "hierarchy",
            )
            corrections = {
                variant: fit_residual_correction(
                    train_groups,
                    valid_groups,
                    residual.copy(),
                    variant,
                )
                for variant in VARIANTS
            }
            for offset, row_index in enumerate(outer_valid):
                row_id = int(train.iloc[row_index].Id)
                record = {
                    "Id": row_id,
                    "seed": seed,
                    "fold": outer_fold,
                    "stage": "discovery",
                    "y_log": float(target_log_by_id[row_id]),
                    "champion_oof_log": float(champion_log_by_id[row_id]),
                }
                for variant, correction in corrections.items():
                    record[f"{variant}_correction"] = float(correction[offset])
                records.append(record)
            print(f"discovery seed={seed} fold={outer_fold}/{OUTER_SPLITS}", flush=True)

    confirmed = pd.read_csv(EXPERIMENTS / "phase5_residual_influence_oof.csv")
    for variant in VARIANTS:
        subset = confirmed.loc[confirmed.variant.eq(variant)]
        if set(subset.seed.unique()) != set(CONFIRMATION_SEEDS):
            raise ValueError(f"Confirmation OOF is missing fixed seeds for {variant}")
    for row in confirmed.itertuples(index=False):
        row_id = int(row.Id)
        records.append(
            {
                "Id": row_id,
                "seed": int(row.seed),
                "fold": int(row.fold),
                "stage": "confirmation",
                "y_log": float(row.y_log),
                "champion_oof_log": float(champion_log_by_id[row_id]),
                "variant": row.variant,
                "all_residuals_correction": float(
                    row.residual_correction if row.variant == "all_residuals" else 0.0
                ),
                "winsorize_2_98_correction": float(
                    row.residual_correction if row.variant == "winsorize_2_98" else 0.0
                ),
            }
        )

    results = pd.DataFrame(records)
    # Discovery records are wide; confirmation records came from a long-form OOF artifact.
    for variant in VARIANTS:
        column = f"{variant}_correction"
        if column not in results:
            results[column] = 0.0
        if variant == "all_residuals":
            mask = results.stage.eq("confirmation") & results.get("variant", pd.Series(index=results.index, dtype=object)).eq(variant)
            results.loc[mask, column] = results.loc[mask, "all_residuals_correction"]
        else:
            mask = results.stage.eq("confirmation") & results.get("variant", pd.Series(index=results.index, dtype=object)).eq(variant)
            results.loc[mask, column] = results.loc[mask, "winsorize_2_98_correction"]

    discovery = results.loc[results.stage.eq("discovery")].copy()
    confirmation = results.loc[results.stage.eq("confirmation")].copy()
    discovery_scores = {}
    confirmation_scores = {}
    for variant in VARIANTS:
        for scale in SCALES:
            key = (variant, scale)
            prediction_column = f"candidate_{variant}_{scale:g}"
            discovery[prediction_column] = discovery.champion_oof_log + scale * discovery[
                f"{variant}_correction"
            ]
            confirmation[prediction_column] = confirmation.champion_oof_log + scale * confirmation[
                f"{variant}_correction"
            ]
            discovery_scores[key] = summarize(discovery, prediction_column)
            confirmation_scores[key] = summarize(confirmation, prediction_column)

    selected = min(discovery_scores, key=lambda config: discovery_scores[config]["pooled_rmse"])
    selected_variant, selected_scale = selected
    selected_column = f"candidate_{selected_variant}_{selected_scale:g}"
    base_confirmation = summarize(confirmation, "champion_oof_log")
    selected_confirmation = confirmation_scores[selected]
    pooled_gain = base_confirmation["pooled_rmse"] - selected_confirmation["pooled_rmse"]
    worst_change = selected_confirmation["worst_fold_rmse"] - base_confirmation["worst_fold_rmse"]
    by_seed = {}
    for seed, rows in confirmation.groupby("seed"):
        by_seed[str(int(seed))] = {
            "base": rmse(rows.y_log.to_numpy(), rows.champion_oof_log.to_numpy()),
            "candidate": rmse(rows.y_log.to_numpy(), rows[selected_column].to_numpy()),
        }
    positive_seed_count = sum(values["candidate"] < values["base"] for values in by_seed.values())
    eligible = pooled_gain >= 0.0003 and worst_change <= 0.01 and positive_seed_count >= 2

    output_oof = EXPERIMENTS / "phase5_winsorized_champion_transfer_oof.csv"
    output_report = EXPERIMENTS / "phase5_winsorized_champion_transfer_validation.json"
    candidate_path = EXPERIMENTS / "shadow_candidates" / "phase5_winsorized_champion_transfer.csv"
    components_path = EXPERIMENTS / "phase5_winsorized_champion_test_components.csv"
    if output_oof.exists() or output_report.exists() or (
        eligible and (candidate_path.exists() or components_path.exists())
    ):
        raise FileExistsError("Refusing to overwrite winsorized champion-transfer artifacts")
    results.to_csv(output_oof, index=False)

    candidate_written = False
    if eligible:
        full_model_preprocessor = phase2.onehot_preprocessor(X)
        train_matrix = full_model_preprocessor.fit_transform(X)
        test_matrix = full_model_preprocessor.transform(X_test)
        full_model = clone(model_spec["model"])
        full_model.fit(train_matrix, y)
        inner_oof = np.zeros(len(X), dtype=float)
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=2031)
        for inner_train, inner_valid in inner.split(X):
            preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
            inner_train_matrix = preprocessor.fit_transform(X.iloc[inner_train])
            inner_valid_matrix = preprocessor.transform(X.iloc[inner_valid])
            model = clone(model_spec["model"])
            model.fit(inner_train_matrix, y[inner_train])
            inner_oof[inner_valid] = model.predict(inner_valid_matrix)
        fit_groups, test_groups = hierarchy.group_frame(train, test, "hierarchy")
        residual = y - inner_oof
        correction = fit_residual_correction(
            fit_groups,
            test_groups,
            residual.copy(),
            selected_variant,
        )
        champion = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12654.csv")
        champion_log = np.log1p(champion.SalePrice.to_numpy(dtype=float))
        candidate_log = champion_log + selected_scale * correction
        candidate_price = np.expm1(candidate_log)
        if not np.isfinite(candidate_price).all() or (candidate_price <= 0).any():
            raise ValueError("Winsorized champion-transfer submission is invalid")
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price}).to_csv(candidate_path, index=False)
        pd.DataFrame(
            {
                "Id": test.Id,
                "champion_log": champion_log,
                "residual_correction_log": correction,
                "scale": selected_scale,
                "candidate_log": candidate_log,
                "champion_price": champion.SalePrice,
                "candidate_price": candidate_price,
            }
        ).to_csv(components_path, index=False)
        candidate_written = True

    report = {
        "experiment_id": "P5-WINSORIZED-CHAMPION-TRANSFER",
        "hypothesis": "A robustly fit, ridge-shrunk hierarchy transfers residual information to the protected champion without depending on extreme residual labels.",
        "champion_oof": "experiments/champion/oof_seed42.csv",
        "correction_oof": "Nested XGBoost residuals; discovery residuals generated in this run; confirmation residual variants from phase5_residual_influence_oof.csv",
        "group_representation": "hierarchy",
        "ridge_alpha": RIDGE_ALPHA,
        "minimum_frequency": MIN_FREQUENCY,
        "winsor_quantiles": [0.02, 0.98],
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "selected_variant_and_scale_on_discovery_only": {
            "variant": selected_variant,
            "scale": selected_scale,
        },
        "discovery_by_config": {
            f"{variant}|scale={scale:g}": discovery_scores[(variant, scale)]
            for variant, scale in discovery_scores
        },
        "confirmation_baseline": base_confirmation,
        "confirmation_candidate": selected_confirmation,
        "confirmation_gain": pooled_gain,
        "confirmation_worst_fold_change": worst_change,
        "confirmation_by_seed": by_seed,
        "confirmation_seeds_improved": positive_seed_count,
        "shadow_gate": "Pooled confirmation gain >=0.0003, worst fold <=baseline+0.01, and at least two of three seeds improve",
        "shadow_gate_passed": bool(eligible),
        "shadow_candidate_written": candidate_written,
        "shadow_candidate_path": str(candidate_path.relative_to(ROOT)) if candidate_written else None,
        "components_path": str(components_path.relative_to(ROOT)) if candidate_written else None,
        "external_champion_score": 0.12654,
        "external_score": None,
        "decision": "SHADOW_ONLY_ROBUST_TRANSFER" if eligible else "REJECT",
        "caveat": "The protected champion OOF is one fixed seed-42 crossfit; corrections are outer-fold cross-fitted. This is not repeated retraining of the full champion and remains unsubmitted.",
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "oof_path": str(output_oof.relative_to(ROOT)),
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
                "change": "Winsorized inner-OOF XGB residual hierarchy transferred to champion OOF; discovery-only scale",
                "cv": f"{selected_confirmation['pooled_rmse']:.6f}",
                "cv_std": f"{selected_confirmation['fold_rmse_std']:.6f}",
                "validation_scheme": "Champion OOF transfer; nested residual folds; discovery/confirmation 3 seeds each",
                "runtime_seconds": report["runtime_seconds"],
                "oof_correlation": "",
                "decision": report["decision"],
                "reason": f"Selected {selected_variant}, scale={selected_scale:g}; gain={pooled_gain:.6f}; fixed champion OOF seed42, no external score",
                "status": report["decision"],
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()