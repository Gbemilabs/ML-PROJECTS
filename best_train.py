"""Reproduce the externally scored historical baseline into isolated best_* files.

This intentionally uses the preserved historical pipeline for score reproduction.
Its split-specific numeric-category string formatting is documented in final_report.md;
use train.py for the corrected preprocessing candidate.
"""

from __future__ import annotations

import importlib.util
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE_SCRIPT = ROOT / "experiments" / "baseline" / "train.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    output_dir = parser.parse_args().output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "best_submission.csv",
        output_dir / "best_experiments.csv",
        output_dir / "best_oof_predictions.csv",
    ]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing_paths)
        )

    spec = importlib.util.spec_from_file_location(
        "preserved_historical_baseline",
        BASELINE_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preserved baseline at {BASELINE_SCRIPT}")

    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)

    baseline.ROOT = ROOT
    baseline.DATA_DIR = ROOT / "data"
    baseline.TRAIN_PATH = baseline.DATA_DIR / "train.csv"
    baseline.TEST_PATH = baseline.DATA_DIR / "test.csv"
    baseline.SUBMISSION_PATH = output_paths[0]
    baseline.EXPERIMENT_PATH = output_paths[1]
    baseline.OOF_PATH = output_paths[2]
    baseline.train_and_submit()


if __name__ == "__main__":
    main()