# Submission Candidates

CSV files here are versioned candidate artifacts. They are not automatically external champions.

## `phase5_support_floor_hierarchy.csv`

- Status: current user-reported external champion; root `submission.csv` and `experiments/champion/submission_0.12374.csv` are byte-identical.
- SHA-256: `52e1309c46960f1d2c411dae3a9148de6db21a00ed264629097edd9cf7a756f8`.
- Method: protected champion log prediction plus a support-pooled Ridge residual hierarchy fit to inner-OOF regularized-XGBoost residuals. Selected settings: hierarchy representation, `alpha=10`, `min_frequency=10`, correction scale `0.5`.
- Validation: three confirmation seeds (2033, 2034, 2035) improved the champion's saved seed-42 OOF pooled RMSE from 0.125466 to 0.124070. Excluding IDs 524 and 1299 still improved 0.117692 to 0.116668. The paired bootstrap was positive in 98.6% of all-row resamples and 94.5% without those rows; outlier-excluded uncertainty still includes zero.
- External result: user-reported RMSLE 0.12374 / approximately rank 759; Kaggle API/submission ID unavailable. Previous champions 0.12654 and 0.127417 remain immutable under `experiments/champion/` and `best_submission.csv`.
- Reproduction source: `experiments/phase5_support_floor_transfer.py`; detailed OOF and validation outputs are under `experiments/phase5_support_floor_transfer_*`.

## `phase6_leaf_similarity_rejected.csv`

- Status: rejected phase-six shadow, not submitted.
- SHA-256: `3e508320111d4a48ce194e5a27c07d9570852a1d0920fcf2219dd48b2416be3a`.
- Method: XGBoost tree-leaf similarity residual neighbors.
- Confirmation OOF improved in aggregate, but excluding the two Edwards cases left only 0.000089 RMSE gain with a paired-bootstrap interval crossing zero. Test ID 2683 moved by +$74,668.
- Original experiment artifact: `experiments/shadow_candidates/phase6_leaf_residual.csv`; full OOF and influence report remain in `experiments/`.

## `phase6_uncertainty_gate_rejected.csv`

- Status: rejected phase-six shadow, not submitted.
- SHA-256: `423a5c9fccf315023df363ad2c9f900daafdc1f8e927ea729ab01136edb7c210`.
- Method: gate an incremental support-pooled residual correction using cross-seed correction dispersion.
- The discovery-selected 90th-percentile gate worsened confirmation champion-OOF RMSE from 0.123932 to 0.124194. Original experiment artifact: `experiments/shadow_candidates/phase6_uncertainty_gate.csv`.
