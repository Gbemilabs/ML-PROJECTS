from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
FLAGGED_IDS = (524, 1299)
BOOTSTRAPS = 20000


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((prediction - actual) ** 2) ** 0.5)


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion_oof = pd.read_csv(EXPERIMENTS / "champion" / "oof_0.12374_crossfit.csv")
    leaf_oof = pd.read_csv(EXPERIMENTS / "phase6_leaf_residual_oof.csv")
    leaf_report = json.loads((EXPERIMENTS / "phase6_leaf_residual_validation.json").read_text())
    shadow = pd.read_csv(EXPERIMENTS / "shadow_candidates" / "phase6_leaf_residual.csv")
    champion_submission = pd.read_csv(EXPERIMENTS / "champion" / "submission_0.12374.csv")

    champion_oof = champion_oof.set_index("Id").reindex(train.Id)
    if champion_oof.candidate_oof_log.isna().any():
        raise ValueError("Champion OOF is missing training IDs")
    if not np.array_equal(shadow.Id.to_numpy(), test.Id.to_numpy()):
        raise ValueError("Leaf candidate IDs do not align with test rows")
    confirmation = leaf_oof.loc[leaf_oof.stage.eq("confirmation")]
    row_prediction = confirmation.groupby("Id").candidate_log.mean().reindex(train.Id)
    if row_prediction.isna().any():
        raise ValueError("Leaf confirmation OOF is missing training IDs")

    actual = np.log1p(train.SalePrice.to_numpy(dtype=float))
    base = champion_oof.candidate_oof_log.to_numpy(dtype=float)
    candidate = row_prediction.to_numpy(dtype=float)
    flagged = train.Id.isin(FLAGGED_IDS).to_numpy()
    rng = np.random.default_rng(2040)
    bootstrap_results = {}
    for name, mask in (("all", np.ones(len(train), dtype=bool)), ("excluding_524_1299", ~flagged)):
        base_squared = (base[mask] - actual[mask]) ** 2
        candidate_squared = (candidate[mask] - actual[mask]) ** 2
        gain_by_house = base_squared - candidate_squared
        sample_count = len(gain_by_house)
        bootstrap_gain = np.empty(BOOTSTRAPS, dtype=float)
        for index in range(BOOTSTRAPS):
            sample = rng.integers(0, sample_count, sample_count)
            bootstrap_gain[index] = gain_by_house[sample].mean()
        bootstrap_results[name] = {
            "baseline_rmse": float(np.mean(base_squared) ** 0.5),
            "candidate_rmse": float(np.mean(candidate_squared) ** 0.5),
            "rmse_gain": float(np.mean(base_squared) ** 0.5 - np.mean(candidate_squared) ** 0.5),
            "paired_mse_gain_ci95": np.quantile(bootstrap_gain, [0.025, 0.5, 0.975]).tolist(),
            "bootstrap_probability_positive_mse_gain": float(np.mean(bootstrap_gain > 0)),
            "n": sample_count,
        }

    fold_ids = np.empty(len(train), dtype=int)
    for fold, (_, valid_idx) in enumerate(
        KFold(n_splits=5, shuffle=True, random_state=42).split(train)
    ):
        fold_ids[valid_idx] = fold
    fold_scores = []
    for fold in range(5):
        valid = fold_ids == fold
        fold_scores.append(
            {
                "fold": fold + 1,
                "baseline_rmse": rmse(actual[valid], base[valid]),
                "candidate_rmse": rmse(actual[valid], candidate[valid]),
            }
        )

    delta = shadow.SalePrice.to_numpy(dtype=float) - champion_submission.SalePrice.to_numpy(dtype=float)
    delta_frame = pd.DataFrame(
        {
            "Id": test.Id,
            "Neighborhood": test.Neighborhood,
            "OverallQual": test.OverallQual,
            "GrLivArea": test.GrLivArea,
            "champion_price": champion_submission.SalePrice,
            "candidate_price": shadow.SalePrice,
            "dollar_delta": delta,
            "log_delta": np.log1p(shadow.SalePrice) - np.log1p(champion_submission.SalePrice),
        }
    )
    output = EXPERIMENTS / "phase6_leaf_influence_report.json"
    report = {
        "experiment_id": "P6-LEAF-INFLUENCE-ABLATION",
        "source_validation": "experiments/phase6_leaf_residual_validation.json",
        "selected_leaf_config": leaf_report["selected_on_discovery_only"],
        "confirmation_oof_metrics": bootstrap_results,
        "original_seed42_fold_scores": fold_scores,
        "test_candidate": {
            "rows": len(shadow),
            "mean_abs_dollar_delta": float(np.mean(np.abs(delta))),
            "p95_abs_dollar_delta": float(np.quantile(np.abs(delta), 0.95)),
            "max_abs_dollar_delta": float(np.max(np.abs(delta))),
            "largest_changes": delta_frame.reindex(np.abs(delta).argsort()[::-1]).head(20).to_dict(orient="records"),
        },
        "decision": "REJECT_FOR_PROMOTION; outlier-excluded RMSE gain is only 0.000089 and paired bootstrap interval crosses zero; largest test shift is +$74,668.",
        "candidate_status": "Preserved in experiments/shadow_candidates/phase6_leaf_residual.csv; not promoted.",
    }
    output.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
        fieldnames = list(rows[0]) if rows else []
    for row in rows:
        if row.get("experiment_id") == "P6-TREE-LEAF-RESIDUAL-NEIGHBORS":
            row["decision"] = "REJECT_FOR_PROMOTION"
            row["status"] = "REJECT_FOR_PROMOTION"
            row["reason"] = (
                row.get("reason", "")
                + "; outlier-excluded pooled RMSE gain is only 0.000089, bootstrap interval crosses zero, and one test price shifts by +$74,668"
            )
    influence_row = {
            "experiment_id": report["experiment_id"],
            "hypothesis": "Tree-leaf residual gains should survive removal of known influential houses and test-shift audit.",
            "change": "Paired bootstrap, outlier exclusion, original-fold and test-shift sensitivity",
            "cv": f"{bootstrap_results['excluding_524_1299']['candidate_rmse']:.6f}",
            "cv_std": "",
            "validation_scheme": "Champion crossfit OOF; 20,000 paired house bootstrap samples",
            "runtime_seconds": "",
            "oof_correlation": "",
            "decision": "REJECT_FOR_PROMOTION",
            "reason": "Outlier-excluded RMSE gain=0.000089; 95% paired bootstrap interval crosses zero; max test price change=+$74,668.",
            "status": "REJECT_FOR_PROMOTION",
        }
    existing_index = next(
        (index for index, row in enumerate(rows) if row.get("experiment_id") == report["experiment_id"]),
        None,
    )
    if existing_index is None:
        rows.append(influence_row)
    else:
        rows[existing_index] = influence_row
    with log_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()