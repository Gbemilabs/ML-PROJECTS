# Research Timeline

This timeline follows the preserved artifacts and experiment-log order. The repository's Git history has a single initial commit and does not provide reliable per-experiment timestamps; chronology below is based on the actual saved outputs and IDs in `experiments/research_log.csv`, not inferred Git dates.

## 1. Historical Baseline

- The original solution used log-price regression with 70% CatBoost and 30% XGBoost, plus domain features for areas, age, bathrooms, porches, and amenities.
- The saved 5-fold seed-42 mean fold RMSE is 0.124471 (std 0.015763); pooled OOF RMSE is 0.125466.
- Its externally reported score was 0.127417 at approximately rank 1188. The submission and OOF files are preserved under `experiments/baseline/` and `best_submission.csv`.

## 2. Model Screening and Ensemble Search

- `experiments/phase2.py` screened CatBoost, XGBoost, LightGBM, randomized trees, GradientBoosting, regularized linear models, and SVR.
- OOF predictions and prediction/residual correlations were retained in `oof_candidates.csv`, `prediction_correlation.csv`, and `error_correlation.csv`.
- A nested blend of the baseline, regularized XGBoost, and GradientBoosting reached mean fold RMSE 0.123571 but had a worse worst fold (0.150998) and no external confirmation. It was held.

## 3. Distribution-Shift Audit and Canonicalization

- The near-perfect adversarial AUC (0.999992) was traced to integer-coded categorical serialization, especially `GarageCars` represented as `2` in train and `2.0` in test.
- Canonical nullable-integer category labels reduced exact-pipeline train/test AUC to 0.515209. The old AUC was invalidated as a source-shift signal.
- The full-data canonicalized 70/30 candidate is byte-identical to current root `submission.csv`.

## 4. Outlier and Edwards Archetype Research

- OOF residual inspection found severe errors on Edwards IDs 524 and 1299, both high-quality large new-construction sales. Test ID 2550 is structurally similar.
- Deleting the two rows improved one XGBoost OOF result but more than doubled its prediction for test ID 2550, so global deletion was rejected.
- A fold-local two-row archetype mean looked unusually strong, but it was selected after residual/test inspection and was not promoted. Broader target encoding worsened repeated OOF.

## 5. Current External Champion Reconstruction

- The user clarified that the root canonicalized submission earned public RMSLE 0.12654 / approximately rank 1084; it was frozen at `experiments/champion/submission_0.12654.csv`.
- The phase-five support-pooled residual candidate replaced it externally per the user's new report: 0.12374 / approximately rank 759. Its exact bytes are frozen at `experiments/champion/submission_0.12374.csv`; Kaggle API/submission ID remain unavailable.
- `best_submission.csv` remains the older 0.127417 result.
- Kaggle API credentials and submission ID are unavailable; score/rank are user-confirmed, while file identity is hash-verified.

## 6. Phase-Four Error and Structure Research

- A 451-slice error map was generated from baseline OOF residuals. Large errors cluster in sparse high-quality/high-area groups and some lower-quality neighborhood segments; these are descriptive, not standalone correction evidence.
- A continuous comparable-property kernel failed confirmation (pooled RMSE 0.135398 vs 0.125466 baseline); price-density target forms were rejected by nested selection.
- A nested Huber residual correction modestly improved regularized XGBoost across five fixed confirmation seeds (0.123604 to 0.123211 pooled). It has not been tested against the exact 70/30 champion under matched nested splits, so it remains local-only.
- Neighborhood×quality residual offsets initially appeared strong for XGBoost. The support-floor negative control showed the gain disappears with at least two peers and worsens at higher support; removing the two Edwards rows leaves only 0.000259 gain. The shadow CSV is preserved for audit but rejected for promotion.

## 7. Current State

- Protected external champion and root `submission.csv`: user-reported RMSLE 0.12374 / approximately #759; exact candidate is archived under `experiments/champion/`.
- Previous external champion: RMSLE 0.12654 / approximately #1084, still preserved.
- Historical external result: RMSLE 0.127417 / approximately #1188, preserved separately.
- Phase-five candidate evidence: champion-OOF confirmation pooled RMSE 0.125466 → 0.124070 across three correction seeds; outlier-excluded pooled OOF 0.117692 → 0.116668; test prediction movement mean absolute about $2.1k.
- Phase-six uncertainty and age-quality approaches failed; tree-leaf residual neighbors had a fragile gain and were rejected after influence analysis.
- Champion OOF error mapping found no ID trend; old low-quality residual groups were unstable under nested correction. Global intercept/affine/Ridge/isotonic calibration all degraded confirmation.
- Global champion-output calibration (intercept, affine, Ridge, isotonic) also failed confirmation; identity predictions remained best.
- No newer Kaggle score is available. Highest-value next step is matched repeated/nested retraining of the full champion/correction system, then external verification when Kaggle access is available.

## 8. Phase Seven Candidate and Lineage Audit

- Phase Five transfer code loads the 0.12654 submission/seed-42 OOF, not the current 0.12374 artifact. The current champion CSV is byte-identical to the archived Phase Five candidate and saved test components match exactly. Its OOF equals the fixed parent OOF plus 0.5 times the averaged residual correction, with five correction predictions per row.
- A dual categorical-plus-numeric/count and ordinal-grade view improved outlier-excluded regularized-XGBoost RMSE across three matched seeds; all-row RMSE improved on each, marginally on seed 2039, while the worst fold regressed on seed 42. This remains an XGBoost-only representation lead.
- Generated `submissions/candidates/next_submission_treeleaf_halfstrength.csv` and immutable shadow copy (SHA-256 `962cac25e3ec94a811ae539cf5bbc0133b42033bf6f61ea3ea9a0aee2ca16cc7`). At scale 0.5, all three saved confirmation seeds improve all-row and outlier-excluded RMSE; excluded-row paired-bootstrap uncertainty crosses zero, so status remains `SHADOW`.
- Largest audited move is +$36,131 for NoRidge ID 2683, quality 9 / 3,500 sq ft with one matching neighborhood-quality training example. ID 2550 moves -$14,319 and is close to the known Edwards examples. No public score is claimed; Kaggle API access remains unavailable.