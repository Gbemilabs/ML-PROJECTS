# Ames Housing: Advanced Regression

Research repository for Kaggle's House Prices competition. The goal is to improve the externally evaluated RMSLE through controlled, leakage-safe experiments rather than leaderboard guesses.

## Submission Status

- Root `submission.csv` is the **current user-reported external champion**: RMSLE 0.12374, approximately rank 759/3775. Its immutable archive is `experiments/champion/submission_0.12374.csv`.
- The previous 0.12654 / approximately rank 1084 champion remains frozen at `experiments/champion/submission_0.12654.csv`.
- The historical 0.127417 submission is preserved as `best_submission.csv` and `experiments/baseline/submission.csv`.
- `submissions/submission_registry.csv` records the two protected earlier results, current 0.12374 champion, and shadow/rejected candidates.

## Research Path

1. Historical CatBoost/XGBoost baseline and seed-42 OOF validation.
2. Diverse model screening and nested blending (`experiments/phase2.py`).
3. Serialization artifact discovery, outlier influence, and Edwards archetype research (`phase3_*`).
4. Error mapping, comparables, target-density, and nested residual studies (`phase4_*`).
5. Support-pooled champion residual transfer; user reports the Phase Five CSV earned 0.12374 and is now the current champion (`phase5_*`).
- Root `submission.csv` is the **current user-reported external champion**: RMSLE 0.12374, approximately rank 759/3775. Its immutable archive is `experiments/champion/submission_0.12374.csv`.
6. Uncertainty gating, age/quality residual interactions, and tree-leaf residual neighbors (`phase6_*`); all phase-six candidates were rejected for promotion.

See `research_timeline.md` for chronology, `research_frontier.md` for the current evidence map, `research_state.md` for the handoff, and `final_report.md` for methods and results.

## Validation and Leakage Controls

The 0.12374 champion has an archived crossfit OOF reconstruction. The phase-five support-pooled correction was selected on discovery seeds and checked on separate nested confirmation seeds; residual targets come from inner-fold OOF XGBoost predictions. The score/rank are user-reported; Kaggle API access is unavailable for independent verification.

Numeric-coded categorical values are canonicalized consistently in train and test. The earlier adversarial AUC of 0.999992 was a serialization artifact; after correction, AUC was approximately 0.515. Do not use the old AUC for reweighting.

## Reproduction

- Data and schema: `data/train.csv`, `data/test.csv`, `data/data_description.txt`.
- Current reported champion: `submission.csv`; immutable version: `experiments/champion/submission_0.12374.csv`.
- Previous champion: `experiments/champion/submission_0.12654.csv`.
- Phase-five nested transfer: `experiments/phase5_support_floor_transfer.py`; its candidate is versioned under `submissions/candidates/`.
- Phase-six uncertainty and tree-leaf experiments are under `experiments/phase6_*`; their generated CSVs remain rejected shadows.
- Phase-six uncertainty and tree-leaf experiments are under `experiments/phase6_*`; their generated CSVs remain rejected shadows. The leaf candidate is versioned at `submissions/candidates/phase6_leaf_similarity_rejected.csv`.
- A new Phase Seven tree-leaf half-strength candidate is ready at `submissions/candidates/next_submission_treeleaf_halfstrength.csv` (SHA-256 `962cac25e3ec94a811ae539cf5bbc0133b42033bf6f61ea3ea9a0aee2ca16cc7`). It remains `SHADOW`; details and test movement audit are in `experiments/phase7_treeleaf_halfstrength_report.json`.
- Phase Five validation uses the 0.12654 parent, while the current 0.12374 OOF adds half of five-mean residual corrections to that fixed parent OOF; repeated full-champion refits remain unresolved.
- Full experiment decisions: `experiments/research_log.csv`.

The scripts refuse to overwrite many existing phase outputs. Preserve champion archives before rerunning experiments. Kaggle CLI/API credentials were unavailable at the latest checkpoint, so phase-six shadow scores are pending external verification.