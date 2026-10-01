# Champion Ledger

This ledger separates user-confirmed public results from API-verifiable results. The current user-reported result is 0.12374 / approximately #759. Its exact CSV is frozen at `experiments/champion/submission_0.12374.csv`, SHA-256 `52e1309c46960f1d2c411dae3a9148de6db21a00ed264629097edd9cf7a756f8`, and is byte-identical to root `submission.csv`. The earlier 0.12654 file remains immutable. Kaggle API verification and submission IDs remain unavailable.

| Version | Public RMSLE | Approx. rank | Artifact / hash | Recipe | Status |
|---|---:|---:|---|---|---|
| Historical champion | 0.127417 | ~1188/3775 | `best_submission.csv`; SHA-256 prefix `ca7ece73455fb26e`; identical to `experiments/baseline/submission.csv` | 70% CatBoost + 30% XGBoost, log target, original engineered features | Preserved historical result |
| Previous external champion | 0.12654 | ~1084/3775 | `experiments/champion/submission_0.12654.csv`; SHA-256 `d2b9ca662c95e40e88e74f32ea1f5970181c6c7614f3fc40d06399c6d1a14008` | Canonicalized 70/30 CatBoost/XGBoost | Preserved; superseded externally per user report |
| Current external champion | 0.12374 | ~759/3775 | `experiments/champion/submission_0.12374.csv`; SHA-256 `52e1309c46960f1d2c411dae3a9148de6db21a00ed264629097edd9cf7a756f8`; byte-identical to root | Support-pooled Ridge hierarchy correction on champion log predictions; see `metadata_0.12374.json` | Current user-reported champion; not queried through Kaggle API |
| Historical external result | 0.127417 | ~1188/3775 | `best_submission.csv`; SHA-256 prefix `ca7ece73455fb26e`; identical to `experiments/baseline/submission.csv` | Historical 70/30 CatBoost/XGBoost system | Preserved historical score |

## Protection Rules

- The 0.12374 score/rank is the active promotion threshold. The score/file association is user-reported; Kaggle submission ID and API query are unavailable.
- `experiments/champion/submission_0.12374.csv`, `experiments/champion/oof_0.12374_crossfit.csv`, and prior champion versions are immutable references. Verify hashes before using them.
- The old 0.127417 submission remains byte-preserved as `best_submission.csv` and `experiments/baseline/submission.csv`.
- Root `submission.csv` is byte-identical to the current 0.12374 champion archive. Phase-six experiments must write to shadow paths until promotion is justified.
- Shadow candidates belong under `experiments/shadow_candidates/`; never replace the protected historical files with a research result.
- For every future champion version, record its SHA-256, row/ID checks, Kaggle submission ID when available, score, rank, model/version, and matching code/artifact set before promotion.

## Verification Limitation

As of this checkpoint, the Kaggle CLI and credentials are absent. The user reports that the Phase Five root candidate earned 0.12374 / approximately #759; its bytes are frozen as `experiments/champion/submission_0.12374.csv`. The 0.12654 and 0.127417 submissions remain separately preserved as prior champions.