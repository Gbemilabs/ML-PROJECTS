# External Champion Archive

Champion files in this directory are immutable. The score/rank history is user-reported; Kaggle API credentials and submission IDs are unavailable for independent queries.

## Current Champion

- Public score: **0.12374 RMSLE**, approximately **rank 759 / 3775**.
- Artifact: `submission_0.12374.csv`, SHA-256 `52e1309c46960f1d2c411dae3a9148de6db21a00ed264629097edd9cf7a756f8`.
- It is byte-identical to root `submission.csv` at the time of this archive and to the phase-five versioned candidate.
- OOF artifact: `oof_0.12374_crossfit.csv`, SHA-256 `2a43378ce8d26aacb1f2286a00b83d48b6afea68e45c4971129e422ce1ee3518`.
- Recipe: previous canonicalized CatBoost/XGBoost champion plus a support-pooled Ridge hierarchy residual correction; hierarchy one-hot minimum frequency 10, Ridge alpha 10, correction scale 0.5, log-price space.
- Local transfer validation: champion OOF pooled RMSE 0.125466 to 0.124070 across three correction seeds; outlier-excluded OOF 0.117692 to 0.116668. The external score is user-reported.

## Previous Champion: 0.12654

- Artifact: `submission_0.12654.csv`, SHA-256 `d2b9ca662c95e40e88e74f32ea1f5970181c6c7614f3fc40d06399c6d1a14008`.
- OOF artifact: `oof_seed42.csv`, SHA-256 `86f20bf831018f8c3f83d4e87ce19a6e51b8aa373bbb32cd6c1746feb8604ab3`.
- Recipe: canonicalized 70% CatBoost + 30% XGBoost, `log1p(SalePrice)`, 3500 CatBoost iterations and 3000 XGBoost estimators; seed-42 KFold mean fold RMSE 0.124471 and pooled OOF RMSE 0.125466.
- This file is preserved as the parent of the phase-five correction; it is no longer the latest reported public score.

## Historical Champion: 0.127417

Preserved separately at `best_submission.csv` and `experiments/baseline/submission.csv` (SHA-256 begins `ca7ece73455fb26e`).

Do not overwrite these archive files. New experiments and shadows belong elsewhere.