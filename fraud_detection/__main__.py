"""Run with `python -m fraud_detection --help` from the repository root."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from fraud_detection.data import DEFAULT_DATA, FEATURES, load_data
from fraud_detection.models import MODEL_NAMES, run_experiment


def main(argv: list[str] | None = None) -> int:
    """Read a CSV, print metrics, and save a reproducible experiment record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Transaction CSV path")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-size", type=int, default=10_000, help="Default: 10000 rows")
    sampling.add_argument("--all-rows", action="store_true", help="Use every row; KNN may be slow")
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Holdout fraction (default: 0.25)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics.json"))
    args = parser.parse_args(argv)
    try:
        frame = load_data(args.data)
        sample_size = None if args.all_rows else args.sample_size
        experiment = run_experiment(
            frame,
            sample_size=sample_size,
            test_size=args.test_size,
            seed=args.seed,
            model_names=tuple(args.models),
        )
        split = experiment.split
        with args.data.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        report = {
            "dataset": {"file": args.data.name, "sha256": digest, "rows": len(frame)},
            "configuration": {
                "sample_size": sample_size,
                "sample_rows": len(split.X_train) + len(split.X_test),
                "test_size": args.test_size,
                "seed": args.seed,
                "features": list(FEATURES),
                "positive_class": 1,
                "confusion_matrix_layout": "rows=actual, columns=predicted; class order=[0,1]",
                "threshold": "Each estimator's default predict rule; no threshold tuning",
            },
            "split": {
                "train_rows": len(split.X_train),
                "test_rows": len(split.X_test),
                "train_class_counts": split.y_train.value_counts().sort_index().to_dict(),
                "test_class_counts": split.y_test.value_counts().sort_index().to_dict(),
            },
            "versions": {
                "python": platform.python_version(),
                **{
                    package: version(package)
                    for package in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")
                },
            },
            "results": experiment.details,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Training rows: {len(split.X_train):,}; holdout rows: {len(split.X_test):,}")
    print(experiment.metrics.round(4).to_string())
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
