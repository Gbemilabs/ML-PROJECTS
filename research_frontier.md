# Research Frontier

**Updated:** 2026-09-30  
**External frontier:** user-reported public RMSLE 0.12374, approximately rank 759/3775, preserved at `experiments/champion/submission_0.12374.csv`; root is byte-identical. Previous 0.12654 and 0.127417 champion files remain archived. See `champion_ledger.md`.

## PROVEN

| Research area | Evidence | Confidence | What remains unknown |
|---|---|---|---|
| Historical baseline | 70% CatBoost + 30% XGBoost; seed-42 5-fold mean fold RMSE 0.124471, std 0.015763, pooled OOF RMSE 0.125466; public score 0.127417/rank ~1188 recorded in project | High for saved local artifacts; external score is previously reported | Whether its local split reproduces hidden-test performance consistently |
| Current external champion identity | `experiments/champion/submission_0.12374.csv`; SHA-256 `52e1309c46960f1d2c411dae3a9148de6db21a00ed264629097edd9cf7a756f8`; byte-identical to root | High for exact file hash and user-reported score; no Kaggle API query | Submission ID and independent API verification |
| GarageCars canonicalization | Exact adversarial pipeline fell from AUC 0.999992 to 0.515209 after canonical integer string conversion; train-side transformed values match historical preprocessing | High | Public-score effect not measured |
| Existing model screen | OOF matrix covers XGBoost, LightGBM, ExtraTrees, RandomForest, GradientBoosting, Ridge variants, ElasticNet, SVR, and baseline | High for saved OOF | Current external champion's OOF/predictions are unavailable |
| Broad target mean | Neighborhood × quality × sale type/condition × build-era shrinkage worsened repeated pooled OOF for all tested strengths; best 0.129313 vs 0.125466 | High for this tested representation | Other local structures remain untested |

## PROMISING

| Research area | Evidence | Confidence | What remains unknown |
|---|---|---|---|
| Local residual market structure | Existing OOF residuals vary across neighborhood/quality/size and two large Edwards cases expose a failure regime | Low to moderate; descriptive and small groups | Whether a more continuous comparable-property model generalizes |
| Nested Huber residual correction | Five-seed nested XGBoost confirmation pooled RMSE 0.123604 to 0.123211 (gain 0.000393); worst fold worsened 0.00117 | Promising only for XGBoost, modest effect | Needs same-fold comparison against the 70/30 champion; do not promote based on XGBoost-only evidence |
| Support-pooled hierarchy transfer | User-reported public RMSLE 0.12374 / ~#759; local champion-OOF transfer 0.125466 to 0.124070 across three confirmation seeds | Current external champion by user report; local validation uses fixed parent seed-42 OOF plus cross-fitted correction | Submission ID/API query and repeated full-model refits unavailable |
| Baseline + XGBoost + GradientBoosting | Nested blend mean fold RMSE 0.123571 vs 0.124471, but std/worst fold worsened | Low; 5 fixed folds, selection on same OOF | Repeated/nested blend confirmation and champion comparison |
| Edwards new-construction specialist | Two examples and one test analogue; exploratory correction reduced repeated OOF error | Very low; selected after residual/test inspection | Independent support, nested selection, and external value |
| Half-strength tree-leaf residual shadow | Confirmation row-averaged RMSE 0.123932 → 0.123106 overall and 0.116487 → 0.116265 excluding IDs 524/1299; all three seeds improve at scale 0.5 | Low-to-moderate; saved confirmation only, fixed parent OOF; outlier-excluded bootstrap interval crosses zero | Repeated full-champion refits and external score; test ID 2683 moves +$36.1k |
| Dual numeric/ordinal feature view | Adding numeric views while retaining categories improved outlier-excluded XGBoost RMSE across seeds 42/2038/2039; all-row benefit was mixed | Low; three-seed XGBoost-only screen and one worst-fold regression | Fresh nested confirmation against the full champion |

## UNRESOLVED

| Research area | What is known | Confidence | What remains unknown |
|---|---|---|---|
| Current champion repeated validation | Current champion OOF is a crossfit reconstruction across five phase-five correction seeds over a fixed parent seed-42 OOF | Moderate | Repeated/nested refits of the full champion pipeline and exact Kaggle submission ID |
| Exact parent champion refit | 0.12374 test CSV exactly matches the saved Phase Five candidate/components; OOF equals the fixed 0.12654 seed-42 parent OOF plus averaged half-scale correction | High for saved-artifact lineage; no fresh end-to-end reproduction in this session | Repeated/nested refits of canonicalized CatBoost/XGBoost parent and residual correction on identical outer folds |
| Residual correction vs champion | Nested Huber and hierarchy trials use regularized XGBoost, not the 70/30 champion | High | Whether any correction transfers to the champion ensemble |
| Error families | Large residuals and subgroup summaries exist for baseline | Moderate for diagnostics | Stable error families under independent folds and current external champion |
| Train/test shift | The near-perfect AUC was an artifact; corrected AUC is near chance | High | Adversarial classifier power for nonlinear interactions after corrected inputs |
| Comparable-property value | 72 settings tested with discovery/confirmation seeds; confirmation pooled RMSE 0.135398 vs base 0.125466 | High for rejection of this kernel | Other comparables representations remain possible |

## REJECTED

- Broad neighborhood/quality/sale target means: degraded repeated pooled OOF.
- Global deletion of IDs 524 and 1299: improved XGBoost OOF on ordinary cases but more than doubled the close test analogue estimate.
- Promoting the two-row Edwards target correction: severe feature-selection and sampling risk.
- Promoting the fixed-fold three-model blend: small apparent gain with worse worst fold and no external result.
- Distribution reweighting from the old AUC 0.999992: the signal was serialization, not covariate shift.
- Continuous neighborhood-gated comparable-price kernel: selected discovery model worsened confirmation pooled RMSE by 0.00993.
- Price-density target reparameterization: nested selection chose direct log price in 24/25 outer folds; one area choice was a severe failure.
- Neighborhood×quality residual offset as a general effect: apparent five-seed pooled gain 0.00170 disappears at a two-peer support floor; without IDs 524/1299 only 0.000259 remains. The test shadow shifts ID 2550 materially and is not promoted.
- Phase-six uncertainty-gated correction: discovery-selected gate worsened confirmation RMSE 0.123932 to 0.124194.
- Global intercept, affine, Ridge, and isotonic calibration all worsened confirmation RMSE; identity remained best.
- Phase-six tree-leaf residual neighbors: aggregate gain 0.001295, but outlier-excluded gain only 0.000089 with paired-bootstrap interval crossing zero; one test house changes +$74,668.
- Phase-six quality/age residual hierarchy: confirmation RMSE worsened 0.123932 to 0.124438; discovery selected no age interaction.
- Full-strength tree-leaf correction: ordinary-row gain is not supported by paired bootstrap; keep the separately generated half-strength file as `SHADOW`, not `CHALLENGER`.

## INVALIDATED

- Treating `GarageCars` train labels like `2` and test labels like `2.0` as genuine source separation. This was a preprocessing artifact.

## NOT YET EXTERNALLY TESTED

- All shadow candidates, including `experiments/shadow_candidates/hierarchical_residual_xgb.csv`.
- Phase-six uncertainty and tree-leaf shadow CSVs are rejected and unsubmitted.
- `submissions/candidates/next_submission_treeleaf_halfstrength.csv` is a new, valid Phase Seven shadow with no external score; its test shifts and nearest peers are audited in `experiments/phase7_treeleaf_halfstrength_report.json`.
- The local Huber residual result has not been compared against a repeated/nested 70/30 champion OOF.
- Current champion repeated-seed OOF is not yet available.

## Next Research

Protect the 0.12374 champion. The next high-value gap is repeated/nested refitting of the full champion pipeline and residual corrections on matched folds; current phase-six branches do not justify promotion. Submit a future candidate only after matched validation and when Kaggle access is available.