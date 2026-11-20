from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bootstrap_accuracy(
    values: list[int],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    estimates = array[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(array.mean()), float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linking-review",
        default="outputs/evaluation_full_clean_v2/linking_item_review_v2.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reviewer_analysis_v2",
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()

    rows = read_rows(Path(args.linking_review))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
    ]
    output_rows: list[list[object]] = []
    for subset_name, flag in [
        ("ambiguous", "1"),
        ("unambiguous", "0"),
    ]:
        for model_index, model in enumerate(models):
            values = [
                int(row["correct"])
                for row in rows
                if row["model"] == model
                and row["variant"] == "candidate_constrained"
                and row["ambiguous_surface_in_train_dev"] == flag
            ]
            mean, low, high = bootstrap_accuracy(
                values,
                seed=args.seed + model_index,
                resamples=args.resamples,
            )
            output_rows.append(
                [
                    model,
                    subset_name,
                    mean,
                    low,
                    high,
                    len(values),
                    args.resamples,
                    args.seed + model_index,
                ]
            )

    with (output_dir / "linking_subset_accuracy_cis.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "subset",
                "accuracy",
                "ci_low",
                "ci_high",
                "n",
                "resamples",
                "seed",
            ]
        )
        writer.writerows(output_rows)

    sensitivity = [
        ["failure_category", "3_point", "5_point", "10_point"],
        ["Overall exact-span extraction", "Hybrid", "Model-dominant", "Model-dominant"],
        ["Unsupported entity generation", "Hybrid", "Hybrid", "Model-dominant"],
        ["Entity-boundary error", "Model-dominant under tested intervention", "Model-dominant under tested intervention", "Unresolved (architecture adverse)"],
        ["Candidate-ranking error", "Architecture-dominant", "Architecture-dominant", "Architecture-dominant"],
        ["Malformed ontology code", "Architecture-dominant", "Architecture-dominant", "Unresolved"],
        ["Ambiguous surface-form linking", "Architecture-dominant", "Architecture-dominant", "Architecture-dominant"],
    ]
    with (output_dir / "threshold_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        csv.writer(handle).writerows(sensitivity)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
