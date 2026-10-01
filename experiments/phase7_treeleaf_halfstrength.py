from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

import phase2
from phase6_leaf_residual import leaf_similarity, local_residual_prediction

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CHAMPION_SHA256 = "52e1309c46960f1d2c411dae3a9148de6db21a00ed264629097edd9cf7a756f8"
CORRECTION_SCALE = 0.5
OUTPUTS = (
    ROOT / "submissions" / "candidates" / "next_submission_treeleaf_halfstrength.csv",
    EXPERIMENTS / "shadow_candidates" / "phase7_treeleaf_halfstrength.csv",
    EXPERIMENTS / "phase7_treeleaf_halfstrength_report.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - actual)) ** 0.5)


def confirmation_metrics(train: pd.DataFrame) -> dict:
    oof = pd.read_csv(EXPERIMENTS / "phase6_leaf_residual_oof.csv")
    confirmation = oof.loc[oof.stage.eq("confirmation")].copy()
    if confirmation.empty:
        raise ValueError("Missing phase-six confirmation OOF predictions")

    actual = np.log1p(train.SalePrice.to_numpy(dtype=float))
    flagged = train.Id.isin([524, 1299]).to_numpy()
    per_seed = {}
    fold_scores = []
    for seed, seed_rows in confirmation.groupby("seed", sort=True):
        seed_rows = seed_rows.set_index("Id").reindex(train.Id)
        if seed_rows.candidate_log.isna().any():
            raise ValueError(f"Confirmation OOF does not cover every training ID for seed {seed}")
        base = seed_rows.champion_oof_log.to_numpy(dtype=float)
        correction = seed_rows.residual_correction.to_numpy(dtype=float)
        target = seed_rows.y_log.to_numpy(dtype=float)
        prediction = base + CORRECTION_SCALE * correction
        per_seed[str(int(seed))] = {
            "all_rows_rmse": rmse(target, prediction),
            "excluding_524_1299_rmse": rmse(target[~flagged], prediction[~flagged]),
        }
        fold = seed_rows.fold.to_numpy(dtype=int)
        for fold_id in sorted(np.unique(fold)):
            valid = fold == fold_id
            fold_scores.append(rmse(target[valid], prediction[valid]))

    by_id = confirmation.groupby("Id").agg(
        actual_log=("y_log", "mean"),
        base_log=("champion_oof_log", "mean"),
        correction_log=("residual_correction", "mean"),
    ).reindex(train.Id)
    if by_id.isna().any().any():
        raise ValueError("Averaged confirmation OOF is missing training IDs")
    target = by_id.actual_log.to_numpy(dtype=float)
    base = by_id.base_log.to_numpy(dtype=float)
    candidate = base + CORRECTION_SCALE * by_id.correction_log.to_numpy(dtype=float)
    all_mask = np.ones(len(train), dtype=bool)
    masks = {"all": all_mask, "excluding_524_1299": ~flagged}
    paired = {}
    rng = np.random.default_rng(2041)
    for name, mask in masks.items():
        base_squared = np.square(base[mask] - target[mask])
        candidate_squared = np.square(candidate[mask] - target[mask])
        gain = base_squared - candidate_squared
        bootstrap = np.empty(20000, dtype=float)
        for index in range(len(bootstrap)):
            sample = rng.integers(0, len(gain), len(gain))
            bootstrap[index] = gain[sample].mean()
        paired[name] = {
            "base_rmse": float(np.mean(base_squared) ** 0.5),
            "candidate_rmse": float(np.mean(candidate_squared) ** 0.5),
            "paired_bootstrap_probability_mse_improves": float(np.mean(bootstrap > 0)),
            "paired_bootstrap_mse_gain_ci95": np.quantile(bootstrap, [0.025, 0.5, 0.975]).tolist(),
        }

    return {
        "parent_champion_oof_rmse": rmse(target, base),
        "correction_scale": CORRECTION_SCALE,
        "confirmation_pooled_rmse_over_repeated_rows": rmse(
            confirmation.y_log.to_numpy(dtype=float),
            confirmation.champion_oof_log.to_numpy(dtype=float)
            + CORRECTION_SCALE * confirmation.residual_correction.to_numpy(dtype=float),
        ),
        "confirmation_worst_fold_rmse_over_15_folds": float(max(fold_scores)),
        "confirmation_by_seed": per_seed,
        "row_averaged_confirmation_paired_bootstrap": paired,
        "validation_scope": (
            "Existing phase-six nested correction OOF; scale 0.5 selected on discovery-only "
            "outlier-excluded RMSE. Parent champion OOF is a fixed crossfit reconstruction, "
            "not repeated refits of the full parent ensemble."
        ),
    }


def peer_audit(train: pd.DataFrame, test: pd.DataFrame, row: pd.Series) -> dict:
    numeric = ["GrLivArea", "TotalBsmtSF", "LotArea", "OverallQual", "YearBuilt", "YearRemodAdd", "GarageArea"]
    train_values = train[numeric].astype(float)
    query_values = row[numeric].astype(float)
    scales = (train_values.quantile(0.75) - train_values.quantile(0.25)).replace(0, 1)
    distance = np.sqrt(np.square((train_values - query_values) / scales).mean(axis=1))
    peer_rows = train.assign(distance=distance).nsmallest(5, "distance")
    same_group = train.Neighborhood.eq(row.Neighborhood) & train.OverallQual.eq(row.OverallQual)
    return {
        "neighborhood_quality_training_support": int(same_group.sum()),
        "nearest_training_peers": [
            {
                "Id": int(peer.Id),
                "distance": float(peer.distance),
                "Neighborhood": str(peer.Neighborhood),
                "OverallQual": int(peer.OverallQual),
                "GrLivArea": int(peer.GrLivArea),
                "YearBuilt": int(peer.YearBuilt),
                "SaleCondition": str(peer.SaleCondition),
                "SalePrice": float(peer.SalePrice),
            }
            for peer in peer_rows.itertuples(index=False)
        ],
    }


def main() -> None:
    champion_path = EXPERIMENTS / "champion" / "submission_0.12374.csv"
    root_submission = ROOT / "submission.csv"
    if sha256(champion_path) != CHAMPION_SHA256:
        raise RuntimeError("Protected champion archive hash changed; stopping")
    if sha256(root_submission) != CHAMPION_SHA256 or root_submission.read_bytes() != champion_path.read_bytes():
        raise RuntimeError("Root submission no longer matches the protected champion; stopping")
    existing = [str(path.relative_to(ROOT)) for path in OUTPUTS if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite outputs: " + ", ".join(existing))

    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    champion = pd.read_csv(champion_path)
    if not np.array_equal(champion.Id.to_numpy(), test.Id.to_numpy()):
        raise ValueError("Protected champion IDs do not match test.csv")

    X, X_test, target_series = phase2.prepare_raw(train, test)
    target = target_series.to_numpy(dtype=float)
    spec = phase2.model_specs()["XGBoost_regularized"]
    preprocessor = phase2.onehot_preprocessor(X)
    train_matrix = preprocessor.fit_transform(X)
    test_matrix = preprocessor.transform(X_test)
    full_model = clone(spec["model"]).fit(train_matrix, target)
    train_leaves = full_model.apply(train_matrix)
    test_leaves = full_model.apply(test_matrix)

    inner_oof = np.zeros(len(X), dtype=float)
    inner = KFold(n_splits=4, shuffle=True, random_state=2041)
    for inner_train, inner_valid in inner.split(X):
        fold_preprocessor = phase2.onehot_preprocessor(X.iloc[inner_train])
        inner_train_matrix = fold_preprocessor.fit_transform(X.iloc[inner_train])
        inner_valid_matrix = fold_preprocessor.transform(X.iloc[inner_valid])
        model = clone(spec["model"]).fit(inner_train_matrix, target[inner_train])
        inner_oof[inner_valid] = model.predict(inner_valid_matrix)

    residual = target - inner_oof
    similarity = leaf_similarity(train_leaves, test_leaves)
    correction, effective_neighbors = local_residual_prediction(
        similarity,
        residual,
        top_k=5,
        power=2.0,
        shrinkage=10.0,
    )
    champion_log = np.log1p(champion.SalePrice.to_numpy(dtype=float))
    candidate_log = champion_log + CORRECTION_SCALE * correction
    candidate_price = np.expm1(candidate_log)
    candidate = pd.DataFrame({"Id": test.Id, "SalePrice": candidate_price})

    if list(candidate.columns) != ["Id", "SalePrice"]:
        raise ValueError("Candidate schema is invalid")
    if len(candidate) != 1459 or not np.array_equal(candidate.Id.to_numpy(), test.Id.to_numpy()):
        raise ValueError("Candidate row count or IDs do not match test.csv")
    if candidate.Id.duplicated().any() or candidate.SalePrice.isna().any():
        raise ValueError("Candidate contains duplicate IDs or missing prices")
    if not np.isfinite(candidate.SalePrice.to_numpy()).all() or (candidate.SalePrice <= 0).any():
        raise ValueError("Candidate contains non-finite or non-positive prices")

    candidate_path, archive_path, report_path = OUTPUTS
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(candidate_path, index=False)
    archive_path.write_bytes(candidate_path.read_bytes())
    candidate_hash = sha256(candidate_path)
    if sha256(archive_path) != candidate_hash:
        raise RuntimeError("Versioned shadow copy differs from submission candidate")

    log_delta = CORRECTION_SCALE * correction
    price_delta = candidate_price - champion.SalePrice.to_numpy(dtype=float)
    correlation = float(np.corrcoef(champion_log, candidate_log)[0, 1])
    delta_frame = test.copy()
    delta_frame["champion_price"] = champion.SalePrice.to_numpy(dtype=float)
    delta_frame["candidate_price"] = candidate_price
    delta_frame["price_delta"] = price_delta
    delta_frame["log_delta"] = log_delta
    delta_frame["effective_leaf_neighbors"] = effective_neighbors
    delta_frame["HouseAge"] = test.YrSold - test.YearBuilt
    top_up = delta_frame.nlargest(10, "price_delta")
    top_down = delta_frame.nsmallest(10, "price_delta")

    def mover_records(frame: pd.DataFrame) -> list[dict]:
        records = []
        for _, row in frame.iterrows():
            audit = {
                "Id": int(row.Id),
                "Neighborhood": str(row.Neighborhood),
                "OverallQual": int(row.OverallQual),
                "GrLivArea": int(row.GrLivArea),
                "HouseAge": int(row.HouseAge),
                "SaleCondition": str(row.SaleCondition),
                "champion_price": float(row.champion_price),
                "candidate_price": float(row.candidate_price),
                "price_delta": float(row.price_delta),
                "log_delta": float(row.log_delta),
                "effective_leaf_neighbors": float(row.effective_leaf_neighbors),
            }
            audit.update(peer_audit(train, test, row))
            records.append(audit)
        return records

    confirmation = confirmation_metrics(train)
    report = {
        "experiment_id": "P7-TREE-LEAF-HALF-STRENGTH-CHALLENGE",
        "status": "SHADOW",
        "external_score": None,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "versioned_copy_path": str(archive_path.relative_to(ROOT)),
        "candidate_sha256": candidate_hash,
        "champion_sha256_before_and_after": sha256(champion_path),
        "method": (
            "Protected champion log prediction plus 0.5 times a top-5 XGBoost tree-leaf "
            "similarity residual correction; Ridge-style effective-neighbor shrinkage 10. "
            "Residual targets are full-training log-price minus four-fold inner-OOF "
            "regularized-XGBoost predictions."
        ),
        "reproduction_command": "python experiments/phase7_treeleaf_halfstrength.py",
        "training": {
            "full_fit_model": "XGBoost_regularized",
            "residual_inner_oof": "4-fold KFold, shuffle=True, random_state=2041",
            "leaf_similarity": {"top_k": 5, "power": 2.0, "shrinkage": 10.0},
            "correction_scale": CORRECTION_SCALE,
        },
        "local_validation": confirmation,
        "test_comparison": {
            "prediction_correlation_with_champion": correlation,
            "mean_abs_log_delta": float(np.mean(np.abs(log_delta))),
            "median_abs_log_delta": float(np.median(np.abs(log_delta))),
            "p95_abs_log_delta": float(np.quantile(np.abs(log_delta), 0.95)),
            "max_abs_log_delta": float(np.max(np.abs(log_delta))),
            "mean_price_delta": float(np.mean(price_delta)),
            "max_abs_price_delta": float(np.max(np.abs(price_delta))),
            "top_10_upward_movers": mover_records(top_up),
            "top_10_downward_movers": mover_records(top_down),
        },
        "integrity": {
            "columns": list(candidate.columns),
            "rows": len(candidate),
            "ids_match_test_in_order": True,
            "unique_ids": bool(candidate.Id.is_unique),
            "finite_positive_prices": True,
            "champion_archive_unchanged": True,
            "candidate_and_versioned_copy_byte_identical": True,
        },
        "promotion_decision": (
            "SHADOW: all three saved confirmation seeds improve at scale 0.5, including "
            "after excluding IDs 524 and 1299, but the paired bootstrap interval for the "
            "outlier-excluded MSE gain crosses zero and the full champion pipeline was not "
            "repeated. No public score is claimed."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "candidate_path": report["candidate_path"],
        "candidate_sha256": candidate_hash,
        "local_validation": confirmation,
        "report_path": str(report_path.relative_to(ROOT)),
        "status": report["status"],
    }, indent=2))


if __name__ == "__main__":
    main()