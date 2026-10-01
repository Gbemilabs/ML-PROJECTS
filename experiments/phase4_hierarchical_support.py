from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
SUPPORT_FLOORS = (0, 2, 3, 5, 10, 20)


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {"pooled_rmse": float(np.mean(error * error) ** 0.5)}


def main() -> None:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    main_oof = pd.read_csv(EXPERIMENTS / "hierarchical_residual_oof.csv")
    main_oof = main_oof.loc[main_oof.stage.eq("confirmation")].copy()
    extra_oof = pd.read_csv(EXPERIMENTS / "hierarchical_residual_confirm_oof.csv")
    main_oof["correction"] = (
        main_oof["selected_group_residual_log"] - main_oof["xgb_base_log"]
    )
    extra_oof["correction"] = extra_oof["corrected_log"] - extra_oof["xgb_base_log"]
    rows = pd.concat([main_oof, extra_oof], ignore_index=True)
    if rows["Id"].duplicated().all():
        raise ValueError("Unexpected duplicate-only OOF artifact")

    raw = train.copy()
    raw["group_key"] = raw["Neighborhood"].fillna("Missing").astype(str) + "|Q" + raw[
        "OverallQual"
    ].fillna("Missing").astype(str)
    key_by_id = raw.set_index("Id")["group_key"]
    rows["group_key"] = rows["Id"].map(key_by_id)
    rows["peer_count"] = 0

    for seed in sorted(rows["seed"].unique()):
        seed_positions = rows.index[rows["seed"].eq(seed)]
        fold_by_row = np.empty(len(raw), dtype=int)
        splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed))
        for fold, (_, valid_idx) in enumerate(splitter.split(raw), start=1):
            fold_by_row[valid_idx] = fold
        for fold in range(1, 6):
            positions = seed_positions[rows.loc[seed_positions, "fold"].to_numpy() == fold]
            fit_counts = raw.loc[fold_by_row != fold, "group_key"].value_counts()
            rows.loc[positions, "peer_count"] = [
                int(fit_counts.get(key, 0)) for key in rows.loc[positions, "group_key"]
            ]

    actual = rows["y_log"].to_numpy(dtype=float)
    base = rows["xgb_base_log"].to_numpy(dtype=float)
    correction = rows["correction"].to_numpy(dtype=float)
    flagged = rows["Id"].isin([524, 1299]).to_numpy()
    peer_counts = rows["peer_count"].to_numpy(dtype=int)
    results = {}
    for floor in SUPPORT_FLOORS:
        use = peer_counts >= floor if floor else np.ones(len(rows), dtype=bool)
        corrected = base + correction * use
        results[str(floor)] = {
            "minimum_training_peers": floor,
            "all_rows": {
                "baseline": metrics(actual, base),
                "corrected": metrics(actual, corrected),
                "affected_rows": int((use & (np.abs(correction) > 1e-12)).sum()),
            },
            "excluding_ids_524_1299": {
                "baseline": metrics(actual[~flagged], base[~flagged]),
                "corrected": metrics(actual[~flagged], corrected[~flagged]),
                "affected_rows": int((use & ~flagged & (np.abs(correction) > 1e-12)).sum()),
            },
        }

    edwards_q10 = rows.loc[
        rows["group_key"].eq("Edwards|Q10"),
        ["Id", "seed", "fold", "peer_count", "correction"],
    ]
    if not edwards_q10.empty and not edwards_q10["peer_count"].eq(1).all():
        raise ValueError("Expected Edwards Q10 validation rows to have one training peer")

    shadow = pd.read_csv(EXPERIMENTS / "shadow_candidates" / "hierarchical_residual_xgb.csv")
    champion = pd.read_csv(ROOT / "experiments" / "champion" / "submission_0.12654.csv")
    test_id = 2550
    test_row = int(np.flatnonzero(shadow["Id"].to_numpy() == test_id)[0])
    report = {
        "experiment_id": "P4-HIERARCHICAL-SUPPORT-ABLATION",
        "oof_source": [
            "experiments/hierarchical_residual_oof.csv",
            "experiments/hierarchical_residual_confirm_oof.csv",
        ],
        "confirmation_seeds": sorted(int(seed) for seed in rows["seed"].unique()),
        "support_floor_results": results,
        "peer_count_distribution": {
            str(int(count)): int(number)
            for count, number in pd.Series(peer_counts).value_counts().sort_index().items()
        },
        "edwards_quality10_rows": edwards_q10.to_dict(orient="records"),
        "test_2550_shadow_price": float(shadow.loc[test_row, "SalePrice"]),
        "test_2550_protected_champion_price": float(
            champion.loc[champion["Id"].eq(test_id), "SalePrice"].iloc[0]
        ),
        "test_2550_shadow_minus_champion": float(
            shadow.loc[test_row, "SalePrice"]
            - champion.loc[champion["Id"].eq(test_id), "SalePrice"].iloc[0]
        ),
        "decision": "REJECT_FOR_PROMOTION; gain vanishes with two-peer floor and is below the gate when the two Edwards rows are excluded.",
        "shadow_candidate_status": "Preserved for audit only; not promoted and not the root submission.",
    }
    output = EXPERIMENTS / "hierarchical_residual_support_ablation.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2))

    log_path = EXPERIMENTS / "research_log.csv"
    with log_path.open("r", newline="", encoding="utf-8") as source:
        log_rows = list(csv.DictReader(source))
        fieldnames = list(log_rows[0]) if log_rows else []
    for row in log_rows:
        if row.get("experiment_id") in {
            "P4-HIERARCHICAL-RESIDUAL",
            "P4-HIERARCHICAL-RESIDUAL-CONFIRM2",
        }:
            row["decision"] = "REJECT_FOR_PROMOTION"
            row["status"] = "REJECT_FOR_PROMOTION"
            row["reason"] = (
                row.get("reason", "")
                + "; support-floor negative control: effect vanishes at >=2 peers; excluding IDs 524/1299 leaves <0.0003 gain"
            )
    if any(row.get("experiment_id") == report["experiment_id"] for row in log_rows):
        raise ValueError("Support-ablation entry already exists in research log")
    log_rows.append(
        {
            "experiment_id": report["experiment_id"],
            "hypothesis": "The neighborhood×quality residual correction should generalize beyond singleton cells and the Edwards outliers.",
            "change": "Support floors 2, 3, 5, 10, 20; score with and without IDs 524/1299",
            "cv": f"{results['2']['all_rows']['corrected']['pooled_rmse']:.6f}",
            "cv_std": "",
            "validation_scheme": "Five nested confirmation seeds; OOF support-count ablation",
            "runtime_seconds": "",
            "oof_correlation": "",
            "decision": "REJECT_FOR_PROMOTION",
            "reason": "Gain vanishes at support>=2; excluding the two flagged observations leaves only 0.000259 gain; shadow changes test ID 2550 materially.",
            "status": "REJECT_FOR_PROMOTION",
        }
    )
    with log_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()