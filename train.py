import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = ROOT / "submission.csv"
EXPERIMENT_PATH = ROOT / "experiments.csv"
OOF_PATH = ROOT / "oof_predictions.csv"

CATEGORICALS_AS_STR = {
    "MSSubClass",
    "OverallQual",
    "OverallCond",
    "YrSold",
    "MoSold",
    "KitchenAbvGr",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageCars",
    "Fireplaces",
}

SEMANTIC_MISSING = {
    "Alley": "Missing",
    "BsmtQual": "Missing",
    "BsmtCond": "Missing",
    "BsmtExposure": "Missing",
    "BsmtFinType1": "Missing",
    "BsmtFinType2": "Missing",
    "FireplaceQu": "Missing",
    "GarageType": "Missing",
    "GarageFinish": "Missing",
    "GarageQual": "Missing",
    "GarageCond": "Missing",
    "PoolQC": "Missing",
    "Fence": "Missing",
    "MiscFeature": "Missing",
    "MasVnrType": "Missing",
}


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalSF"] = df["1stFlrSF"] + df["2ndFlrSF"] + df["TotalBsmtSF"]
    df["TotalBath"] = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["TotalPorchSF"] = (
        df["OpenPorchSF"]
        + df["EnclosedPorch"]
        + df["3SsnPorch"]
        + df["ScreenPorch"]
    )
    df["LotFrontagePerLotArea"] = df["LotFrontage"] / df["LotArea"]
    return df


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engineered_features(df)
    for column in df.columns:
        if column in {"Id", "SalePrice"}:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if column in CATEGORICALS_AS_STR:
                values = pd.to_numeric(df[column], errors="coerce")
                df[column] = values.astype("Int64").astype("string").fillna("Missing")
            else:
                df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].replace({np.nan: "Missing"})
            df[column] = df[column].fillna("Missing")
            df[column] = df[column].astype(str)
    for column, value in SEMANTIC_MISSING.items():
        if column in df.columns:
            df[column] = df[column].replace({np.nan: value, "nan": value, "NaN": value})
            df[column] = df[column].fillna(value)
    return df


def make_preprocessor(X: pd.DataFrame):
    categorical_columns = [
        col for col in X.columns if X[col].dtype == object or pd.api.types.is_string_dtype(X[col])
    ]
    numeric_columns = [col for col in X.columns if col not in categorical_columns]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_columns,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ],
        remainder="drop",
    )


def build_xgb_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=1,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )


def build_catboost_model():
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=3500,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )


def evaluate_cv(X_train: pd.DataFrame, y_train: pd.Series):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    xgb_fold_scores = []
    cat_fold_scores = []
    ensemble_fold_scores = []
    ensemble_oof = np.zeros(len(X_train), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(X_train), start=1):
        X_tr = X_train.iloc[train_idx].copy()
        X_va = X_train.iloc[valid_idx].copy()
        y_tr = y_train.iloc[train_idx].copy()
        y_va = y_train.iloc[valid_idx].copy()

        xgb_preprocessor = make_preprocessor(X_tr)
        X_tr_xgb = xgb_preprocessor.fit_transform(X_tr)
        X_va_xgb = xgb_preprocessor.transform(X_va)

        xgb_model = build_xgb_model()
        xgb_model.fit(X_tr_xgb, y_tr)
        xgb_pred = xgb_model.predict(X_va_xgb)
        xgb_score = np.sqrt(mean_squared_error(y_va, xgb_pred))
        xgb_fold_scores.append(xgb_score)

        cat_cols = [col for col in X_tr.columns if X_tr[col].dtype == object or pd.api.types.is_string_dtype(X_tr[col])]
        cat_tr = X_tr.copy()
        cat_va = X_va.copy()
        for col in cat_cols:
            cat_tr[col] = cat_tr[col].astype(str)
            cat_va[col] = cat_va[col].astype(str)

        cat_model = build_catboost_model()
        cat_model.fit(cat_tr, y_tr, cat_features=cat_cols)
        cat_pred = cat_model.predict(cat_va)
        cat_score = np.sqrt(mean_squared_error(y_va, cat_pred))
        cat_fold_scores.append(cat_score)

        ensemble_pred = 0.7 * cat_pred + 0.3 * xgb_pred
        ensemble_score = np.sqrt(mean_squared_error(y_va, ensemble_pred))
        ensemble_fold_scores.append(ensemble_score)
        ensemble_oof[valid_idx] = ensemble_pred

        print(f"Fold {fold_idx}: XGB={xgb_score:.6f} CatBoost={cat_score:.6f} Ensemble={ensemble_score:.6f}")

    mean_score = float(np.mean(ensemble_fold_scores))
    std_score = float(np.std(ensemble_fold_scores))
    return mean_score, std_score, ensemble_fold_scores, ensemble_oof


def train_and_submit():
    train_df, test_df = load_data()
    train_df = preprocess_frame(train_df)
    test_df = preprocess_frame(test_df)

    target = np.log1p(train_df["SalePrice"])
    features = [col for col in train_df.columns if col not in {"SalePrice"}]
    X = train_df[features]
    X_test = test_df[features]

    mean_score, std_score, fold_scores, oof_predictions = evaluate_cv(X, target)

    with EXPERIMENT_PATH.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "experiment_id",
            "model",
            "preprocessing",
            "feature_set",
            "cv_fold_scores",
            "cv_mean",
            "cv_std",
            "notes",
        ])
        writer.writerow([
            "EXP-001",
            "XGB + CatBoost ensemble",
            "one_hot+median_impute+log1p_target",
            "engineered_base",
            json.dumps([float(x) for x in fold_scores]),
            f"{mean_score:.6f}",
            f"{std_score:.6f}",
            "Weight 0.7 CatBoost + 0.3 XGB from OOF validation",
        ])

    pd.DataFrame({"Id": X.index + 1, "SalePrice": np.expm1(oof_predictions)}).to_csv(OOF_PATH, index=False)

    xgb_preprocessor = make_preprocessor(X)
    X_xgb = xgb_preprocessor.fit_transform(X)
    X_test_xgb = xgb_preprocessor.transform(X_test)

    cat_cols = [col for col in X.columns if X[col].dtype == object or pd.api.types.is_string_dtype(X[col])]
    X_cat = X.copy()
    X_test_cat = X_test.copy()
    for col in cat_cols:
        X_cat[col] = X_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)

    final_xgb = build_xgb_model()
    final_xgb.fit(X_xgb, target)
    xgb_test_pred = final_xgb.predict(X_test_xgb)

    final_cat = build_catboost_model()
    final_cat.fit(X_cat, target, cat_features=cat_cols)
    cat_test_pred = final_cat.predict(X_test_cat)

    test_predictions = np.expm1(0.7 * cat_test_pred + 0.3 * xgb_test_pred)

    submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": test_predictions})
    submission.to_csv(SUBMISSION_PATH, index=False)

    print(f"CV RMSE(log1p): mean={mean_score:.6f} std={std_score:.6f}")
    print(f"Saved final submission to {SUBMISSION_PATH}")
    print(f"Saved OOF predictions to {OOF_PATH}")
    print(f"Saved experiment log to {EXPERIMENT_PATH}")


if __name__ == "__main__":
    train_and_submit()
