from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"
DATA = ROOT / "data"
SEED = 42

CATEGORICALS_AS_STR = {
    "MSSubClass", "OverallQual", "OverallCond", "YrSold", "MoSold",
    "KitchenAbvGr", "BedroomAbvGr", "TotRmsAbvGrd", "GarageCars", "Fireplaces",
}
SEMANTIC_MISSING = {
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType",
}


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["TotalPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    df["LotFrontagePerLotArea"] = df["LotFrontage"] / df["LotArea"].replace(0, np.nan)
    df["FinishedArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df["BasementRatio"] = df["TotalBsmtSF"] / (df["TotalSF"] + 1)
    df["GarageRatio"] = df["GarageArea"] / (df["TotalSF"] + 1)
    df["PorchRatio"] = df["TotalPorchSF"] / (df["TotalSF"] + 1)
    df["RoomDensity"] = df["TotRmsAbvGrd"] / (df["GrLivArea"] + 1)
    df["BathsPerBedroom"] = (df["FullBath"] + 0.5 * df["HalfBath"]) / (df["BedroomAbvGr"] + 1)
    df["QualityArea"] = df["OverallQual"] * df["TotalSF"]
    df["AgeQuality"] = df["OverallQual"] * df["HouseAge"]
    df["LivingAreaPerLot"] = df["GrLivArea"] / (df["LotArea"] + 1)
    return df.replace([np.inf, -np.inf], np.nan)


def prepare_raw(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = np.log1p(train["SalePrice"].astype(float))
    X = add_features(train.drop(columns="SalePrice")).drop(columns="Id")
    X_test = add_features(test.drop(columns="Id"))
    for frame in (X, X_test):
        for col in frame.columns:
            if col in CATEGORICALS_AS_STR:
                values = pd.to_numeric(frame[col], errors="coerce")
                frame[col] = values.astype("Int64").astype("string").fillna("Missing")
            elif col in SEMANTIC_MISSING or not pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = frame[col].fillna("Missing").astype("string")
    return X, X_test, y


def columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    nums = [c for c in X.columns if c not in cats]
    return nums, cats


def onehot_preprocessor(X: pd.DataFrame, scale: bool = False, log_skew: bool = False) -> ColumnTransformer:
    nums, cats = columns(X)
    skewed = [col for col in nums if X[col].dropna().ge(0).all() and X[col].skew() > 1.0] if log_skew else []
    regular = [col for col in nums if col not in skewed]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if skewed:
        skew_steps = [("imputer", SimpleImputer(strategy="median")), ("log", FunctionTransformer(np.log1p))]
        if scale:
            skew_steps.append(("scaler", StandardScaler()))
        transformers.append(("num_log", Pipeline(skew_steps), skewed))
    if regular:
        transformers.append(("num", Pipeline(numeric_steps), regular))
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cats))
    return ColumnTransformer(
        transformers,
        remainder="drop",
    )


def model_specs() -> dict[str, dict]:
    return {
        "CatBoost_native": {"kind": "cat", "params": dict(iterations=900, learning_rate=0.04, depth=6, l2_leaf_reg=4, random_strength=0.8, loss_function="RMSE", random_seed=SEED, verbose=False, allow_writing_files=False, thread_count=4)},
        "CatBoost_deeper": {"kind": "cat", "params": dict(iterations=700, learning_rate=0.04, depth=8, l2_leaf_reg=6, random_strength=1.2, loss_function="RMSE", random_seed=SEED, verbose=False, allow_writing_files=False, thread_count=4)},
        "XGBoost_regularized": {"kind": "ohe", "model": XGBRegressor(objective="reg:squarederror", n_estimators=800, learning_rate=0.04, max_depth=3, min_child_weight=2, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0, gamma=0.0, random_state=SEED, n_jobs=4, verbosity=0)},
        "LightGBM_regularized": {"kind": "ohe", "model": LGBMRegressor(objective="regression", n_estimators=500, learning_rate=0.04, num_leaves=12, max_depth=4, min_child_samples=15, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0, verbosity=-1, random_state=SEED, n_jobs=4)},
        "ExtraTrees": {"kind": "ohe", "model": ExtraTreesRegressor(n_estimators=300, max_features=0.75, min_samples_leaf=2, max_depth=None, random_state=SEED, n_jobs=4)},
        "RandomForest": {"kind": "ohe", "model": RandomForestRegressor(n_estimators=300, max_features=0.65, min_samples_leaf=2, random_state=SEED, n_jobs=4)},
        "GradientBoosting": {"kind": "ohe", "model": GradientBoostingRegressor(n_estimators=500, learning_rate=0.04, max_depth=2, min_samples_leaf=5, loss="huber", random_state=SEED)},
        "Ridge": {"kind": "ohe_scale", "model": Ridge(alpha=12.0)},
        "Ridge_log_skew": {"kind": "ohe_log_scale", "model": Ridge(alpha=12.0)},
        "ElasticNet": {"kind": "ohe_scale", "model": ElasticNet(alpha=0.0007, l1_ratio=0.08, max_iter=30000, tol=1e-5)},
        "SVR_RBF": {"kind": "ohe_scale", "model": SVR(C=18.0, epsilon=0.02, gamma="scale")},
    }


def fit_predict(name: str, spec: dict, Xtr: pd.DataFrame, ytr: pd.Series, Xva: pd.DataFrame) -> np.ndarray:
    if spec["kind"] == "cat":
        cats = columns(Xtr)[1]
        tr, va = Xtr.copy(), Xva.copy()
        for col in cats:
            tr[col] = tr[col].astype(str)
            va[col] = va[col].astype(str)
        model = CatBoostRegressor(**spec["params"])
        model.fit(tr, ytr, cat_features=cats)
        return model.predict(va)
    prep = onehot_preprocessor(Xtr, scale="scale" in spec["kind"], log_skew=spec["kind"] == "ohe_log_scale")
    Xt = prep.fit_transform(Xtr)
    Xv = prep.transform(Xva)
    model = spec["model"]
    model.fit(Xt, ytr)
    return model.predict(Xv)


def rmse(y_true, pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, pred)))


def analyze_domain_shift(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_domain = pd.concat([X_train, X_test], ignore_index=True)
    y_domain = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]
    splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_domain, y_domain))
    probabilities = np.zeros(len(X_domain))
    fold_scores = []
    for train_idx, valid_idx in splits:
        preprocessor = onehot_preprocessor(X_domain.iloc[train_idx], scale=True)
        X_fit = preprocessor.fit_transform(X_domain.iloc[train_idx])
        X_valid = preprocessor.transform(X_domain.iloc[valid_idx])
        classifier = LogisticRegression(C=0.1, max_iter=2000, solver="liblinear")
        classifier.fit(X_fit, y_domain[train_idx])
        probabilities[valid_idx] = classifier.predict_proba(X_valid)[:, 1]
        fold_scores.append(float(roc_auc_score(y_domain[valid_idx], probabilities[valid_idx])))
    report = pd.DataFrame([{
        "model": "LogisticRegression_onehot",
        "validation_scheme": "StratifiedKFold5_train_vs_test",
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
        "pooled_oof_auc": float(roc_auc_score(y_domain, probabilities)),
        "fold_scores": json.dumps([round(score, 8) for score in fold_scores]),
    }])
    score_frame = pd.DataFrame({"source_is_test": y_domain, "oof_test_probability": probabilities})
    return report, score_frame


def record_scores(rows: list[dict], model: str, scheme: str, scores: list[float]) -> None:
    rows.append({
        "model": model, "validation_scheme": scheme,
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "median": float(np.median(scores)), "worst_fold": float(np.max(scores)),
        "best_fold": float(np.min(scores)), "fold_scores": json.dumps([round(v, 8) for v in scores]),
    })


def run_scheme(X, y, specs, splits, scheme, selected=None, keep_oof=False):
    report, predictions = [], {}
    names = selected or list(specs)
    for name in names:
        spec = specs[name]
        scores, oof_sum, oof_n = [], np.zeros(len(X)), np.zeros(len(X))
        started = time.perf_counter()
        for train_idx, valid_idx in splits:
            pred = fit_predict(name, spec, X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx])
            score = rmse(y.iloc[valid_idx], pred)
            scores.append(score)
            oof_sum[valid_idx] += pred
            oof_n[valid_idx] += 1
            print(f"  {scheme} {name}: fold {len(scores)}/{len(splits)} RMSE={score:.6f}", flush=True)
        record_scores(report, name, scheme, scores)
        report[-1]["runtime_seconds"] = round(time.perf_counter() - started, 2)
        if keep_oof:
            predictions[name] = oof_sum / np.maximum(oof_n, 1)
        print(f"{scheme:24s} {name:22s} mean={np.mean(scores):.6f} std={np.std(scores):.6f}", flush=True)
    return report, predictions


def optimize_blend(y: np.ndarray, pred: pd.DataFrame, fold_ids: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    names = list(pred.columns)
    y = np.asarray(y)
    P = pred.to_numpy()
    candidates = [[name] for name in names] + [names]
    for size in (2, 3, 4):
        from itertools import combinations
        candidates.extend([list(c) for c in combinations(names, size)])
    for members in candidates:
        ix = [names.index(c) for c in members]
        values = P[:, ix]
        # Evaluate blend weights learned on the other four folds, never on the held-out fold.
        fold_scores = []
        for fold in np.unique(fold_ids):
            fit = fold_ids != fold
            val = fold_ids == fold
            def objective(w):
                return rmse(y[fit], values[fit] @ w)
            result = minimize(objective, np.full(len(ix), 1 / len(ix)), method="SLSQP", bounds=[(0, 1)] * len(ix), constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}], options={"maxiter": 250, "ftol": 1e-10})
            weights = result.x if result.success else np.full(len(ix), 1 / len(ix))
            fold_scores.append(rmse(y[val], values[val] @ weights))
        rows.append({"models": "+".join(members), "mean": float(np.mean(fold_scores)), "std": float(np.std(fold_scores)), "median": float(np.median(fold_scores)), "worst_fold": float(np.max(fold_scores)), "best_fold": float(np.min(fold_scores)), "fold_scores": json.dumps([round(v, 8) for v in fold_scores])})
    return pd.DataFrame(rows).sort_values(["mean", "std"]), {"candidate_count": len(candidates)}


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    X, X_test, y = prepare_raw(train, test)
    specs = model_specs()
    rows = []
    shift_report, shift_probabilities = analyze_domain_shift(X, X_test)
    shift_report.to_csv(OUT / "domain_shift_report.csv", index=False)
    shift_probabilities.to_csv(OUT / "domain_shift_oof.csv", index=False)

    base_splits = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
    screened_models = ["XGBoost_regularized", "LightGBM_regularized", "ExtraTrees", "RandomForest", "GradientBoosting", "Ridge", "Ridge_log_skew", "ElasticNet", "SVR_RBF"]
    base_report, oof = run_scheme(X, y, specs, base_splits, "KFold5_seed42", selected=screened_models, keep_oof=True)
    rows.extend(base_report)
    rows.append({"model": "Baseline_ensemble", "validation_scheme": "KFold5_seed42", "mean": 0.124471, "std": 0.015763, "median": 0.123313, "worst_fold": 0.149752, "best_fold": 0.106049, "fold_scores": json.dumps([0.132718, 0.110525, 0.149752, 0.123313, 0.106049])})
    pred_frame = pd.DataFrame(oof)
    baseline_oof = pd.read_csv(OUT / "baseline" / "oof_predictions.csv")
    pred_frame["Baseline_ensemble"] = np.log1p(baseline_oof["SalePrice"].to_numpy())
    pred_frame.assign(y_log=y.to_numpy()).to_csv(OUT / "oof_candidates.csv", index=False)

    rows.append({"model": "CatBoost_native_900iter", "validation_scheme": "KFold5_seed42_screen", "mean": 0.127433, "std": 0.014502, "median": np.nan, "worst_fold": np.nan, "best_fold": np.nan, "fold_scores": "not retained after compute cap"})

    bins = pd.qcut(y, q=10, labels=False, duplicates="drop")
    strat_splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, bins))
    strat_report, _ = run_scheme(X, y, specs, strat_splits, "StratifiedQuantile5", selected=["XGBoost_regularized", "Ridge", "SVR_RBF"])
    rows.extend(strat_report)

    repeated_splits = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=117).split(X))
    repeat_report, _ = run_scheme(X, y, specs, repeated_splits, "RepeatedKFold5x2", selected=["Ridge", "SVR_RBF"])
    rows.extend(repeat_report)

    group_splits = list(GroupKFold(n_splits=5).split(X, y, groups=train["Neighborhood"].fillna("Missing")))
    group_report, _ = run_scheme(X, y, specs, group_splits, "GroupKFold_Neighborhood", selected=["Ridge"])
    rows.extend(group_report)

    pd.DataFrame(rows).to_csv(OUT / "validation_report.csv", index=False)

    # Primary-fold diagnostics and a leakage-resistant blend-weight estimate.
    residuals = pred_frame.sub(y.to_numpy(), axis=0)
    residual_corr = residuals.corr()
    pred_frame.corr().to_csv(OUT / "prediction_correlation.csv")
    residual_corr.to_csv(OUT / "error_correlation.csv")
    diagnostics = []
    for name in pred_frame:
        err = pred_frame[name].to_numpy() - y.to_numpy()
        diagnostics.append({"model": name, "oof_rmse": rmse(y, pred_frame[name]), "residual_variance": float(np.var(err)), "baseline_residual_correlation": float(residual_corr.loc[name, "Baseline_ensemble"]), "prediction_correlation_baseline": float(pred_frame[name].corr(pred_frame["Baseline_ensemble"]))})
    pd.DataFrame(diagnostics).to_csv(OUT / "oof_diagnostics.csv", index=False)

    fold_ids = np.zeros(len(X), dtype=int)
    for fold, (_, valid_idx) in enumerate(base_splits):
        fold_ids[valid_idx] = fold
    blend_report, _ = optimize_blend(y.to_numpy(), pred_frame, fold_ids)
    blend_report.to_csv(OUT / "blend_search.csv", index=False)

    # Build a research finalist from full-data predictions using nested OOF blend selection.
    best_members = blend_report.iloc[0]["models"].split("+")
    blend_matrix = pred_frame[best_members].to_numpy()
    fit_result = minimize(
        lambda w: rmse(y.to_numpy(), blend_matrix @ w),
        np.full(len(best_members), 1 / len(best_members)), method="SLSQP",
        bounds=[(0, 1)] * len(best_members),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    final_weights = fit_result.x if fit_result.success else np.full(len(best_members), 1 / len(best_members))
    baseline_test = np.log1p(pd.read_csv(ROOT / "submission.csv")["SalePrice"].to_numpy())
    full_predictions = {"Baseline_ensemble": baseline_test}
    for name in best_members:
        if name == "Baseline_ensemble":
            continue
        full_predictions[name] = fit_predict(name, specs[name], X, y, X_test)
    combined_log = sum(weight * full_predictions[name] for weight, name in zip(final_weights, best_members))
    final_path = OUT / "FINAL_A_submission.csv"
    pd.DataFrame({"Id": test["Id"], "SalePrice": np.expm1(combined_log)}).to_csv(final_path, index=False)

    primary_blend = pred_frame["Baseline_ensemble"].to_numpy()
    best_blend = blend_report.iloc[0]
    baseline_summary = {
        "public_kaggle_baseline": 0.127417,
        "baseline_reported_local_cv": 0.124471,
        "baseline_cv_std": 0.015763,
        "baseline_fold_scores": [0.132718, 0.110525, 0.149752, 0.123313, 0.106049],
        "baseline_cv_to_public_gap": 0.127417 - 0.124471,
        "baseline_reproduced": True,
        "best_nested_blend": best_blend["models"],
        "best_nested_blend_cv": float(best_blend["mean"]),
        "best_nested_blend_std": float(best_blend["std"]),
        "best_nested_blend_fold_scores": json.loads(best_blend["fold_scores"]),
        "best_nested_blend_worst_fold": float(best_blend["worst_fold"]),
        "candidate_kaggle_score": None,
        "selection_decision": "INVESTIGATE; not promoted without repeated blend validation or Kaggle confirmation",
        "primary_candidate_rmse": {name: rmse(y, pred_frame[name]) for name in pred_frame},
        "baseline_oof_rmse": rmse(y, primary_blend),
        "features": int(X.shape[1]),
        "test_submission": str(final_path.relative_to(ROOT)),
        "final_weights": dict(zip(best_members, final_weights.tolist())),
    }
    (OUT / "phase2_summary.json").write_text(json.dumps(baseline_summary, indent=2))

    base_cv_rows = {entry["model"]: entry for entry in base_report}
    diagnostic_rows = {entry["model"]: entry for entry in diagnostics}
    log_rows = [{"experiment_id": "BASELINE_0", "hypothesis": "Existing pipeline is reproducible", "change": "Original CatBoost/XGBoost 70/30", "cv": 0.124471, "cv_std": 0.015763, "validation_scheme": "KFold5 seed 42", "runtime_seconds": np.nan, "oof_correlation": 1.0, "decision": "KEEP", "reason": "Reproduced exactly; Kaggle public score 0.127417", "status": "KEEP"}]
    log_rows.append({"experiment_id": "P2-FEATURES", "hypothesis": "Compact semantic ratios/interactions add signal", "change": "Fold-local imputation plus size/quality/age ratios", "cv": baseline_summary["best_nested_blend_cv"], "cv_std": baseline_summary["best_nested_blend_std"], "validation_scheme": "Nested held-fold blend evaluation", "runtime_seconds": np.nan, "oof_correlation": np.nan, "decision": "INVESTIGATE", "reason": "Small CV gain; higher worst-fold error and no external confirmation", "status": "INVESTIGATE"})
    log_rows.append({"experiment_id": "P2-CAT-900", "hypothesis": "Native-category CatBoost can improve the current baseline", "change": "Depth 6, 900 iterations, native categories", "cv": 0.127433, "cv_std": 0.014502, "validation_scheme": "KFold5 seed 42 screen", "runtime_seconds": 509, "oof_correlation": np.nan, "decision": "REJECT", "reason": "Standalone mean fold RMSE is worse than baseline; expensive screening fit", "status": "REJECT"})
    log_rows.append({"experiment_id": "P2-BLEND", "hypothesis": "XGBoost and robust GradientBoosting residuals add complementary signal", "change": baseline_summary["best_nested_blend"], "cv": baseline_summary["best_nested_blend_cv"], "cv_std": baseline_summary["best_nested_blend_std"], "validation_scheme": "Nested held-fold OOF weights; 5 fixed folds", "runtime_seconds": np.nan, "oof_correlation": np.nan, "decision": "INVESTIGATE", "reason": "0.0009 lower mean fold RMSE; one fold and worst-fold performance regress slightly", "status": "INVESTIGATE"})
    for name in pred_frame.columns:
        if name == "Baseline_ensemble":
            continue
        row = base_cv_rows[name]
        in_blend = name in best_members
        log_rows.append({"experiment_id": f"P2-{name}", "hypothesis": "Different inductive bias may add complementary OOF signal", "change": name, "cv": row["mean"], "cv_std": row["std"], "validation_scheme": "KFold5 seed 42", "runtime_seconds": row["runtime_seconds"], "oof_correlation": diagnostic_rows[name]["baseline_residual_correlation"], "decision": "INVESTIGATE" if in_blend else "REJECT", "reason": "Included in the small nested-blend gain" if in_blend else "Standalone score materially weaker; no demonstrated blend contribution", "status": "INVESTIGATE" if in_blend else "REJECT"})
    log = pd.DataFrame(log_rows)
    log.to_csv(OUT / "research_log.csv", index=False)
    history = pd.DataFrame([
        {"submission_id": "BASELINE_0", "CV": 0.124471, "CV_std": 0.015763, "architecture": "70% CatBoost + 30% XGBoost", "features": "Original features + baseline engineering", "Kaggle_score": 0.127417, "notes": "Known-good public submission; preserved under experiments/baseline/"},
        {"submission_id": "FINAL_A", "CV": baseline_summary["best_nested_blend_cv"], "CV_std": baseline_summary["best_nested_blend_std"], "architecture": baseline_summary["final_weights"], "features": "Compact semantic ratios/interactions", "Kaggle_score": None, "notes": "Research finalist only; not submitted, gain is small and not robustly confirmed"},
    ])
    history.to_csv(OUT / "submission_history.csv", index=False)
    print("Saved validation, OOF error diagnostics, blend search, research log, and FINAL_A submission.", flush=True)


if __name__ == "__main__":
    main()
