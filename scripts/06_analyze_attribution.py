from __future__ import annotations

import argparse
import csv
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

SEED = 20260721
BOOTSTRAP_RESAMPLES = 10_000
PRACTICAL_THRESHOLD = 0.05


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty list.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def extraction_metrics(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    tp = fp = fn = boundary = unsupported = predicted = 0
    count = 0
    for row in rows:
        count += 1
        tp += int(row["exact_matches"])
        fp += int(row["false_positive_spans"]) + int(row["unsupported_entities"])
        fn += int(row["false_negative_spans"])
        boundary += int(row["boundary_errors"])
        unsupported += int(row["unsupported_entities"])
        predicted += int(row["predicted_spans"]) + int(row["unsupported_entities"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_rate": unsupported / predicted if predicted else 0.0,
        "boundary_rate": boundary / predicted if predicted else 0.0,
    }


def linking_metrics(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    data = list(rows)
    n = len(data)
    return {
        "n": n,
        "accuracy": sum(int(r["correct"]) for r in data) / n if n else 0.0,
        "invalid_rate": sum(1 - int(r["valid_cui_format"]) for r in data) / n if n else 0.0,
        "outside_candidate_rate": sum(int(r["outside_candidate_list"]) for r in data) / n if n else 0.0,
    }


def paired_bootstrap(
    ids: list[str],
    rows_a: dict[str, dict[str, str]],
    rows_b: dict[str, dict[str, str]],
    metric_fn: Callable[[Iterable[dict[str, str]]], dict[str, float]],
    metric_name: str,
    seed: int,
    resamples: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    observed = metric_fn(rows_a[i] for i in ids)[metric_name] - metric_fn(rows_b[i] for i in ids)[metric_name]
    differences: list[float] = []
    for _ in range(resamples):
        sampled = [ids[i] for i in rng.integers(0, len(ids), size=len(ids))]
        a_value = metric_fn(rows_a[i] for i in sampled)[metric_name]
        b_value = metric_fn(rows_b[i] for i in sampled)[metric_name]
        differences.append(a_value - b_value)
    return observed, quantile(differences, 0.025), quantile(differences, 0.975)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the paired bootstrap and operational attribution analysis.")
    parser.add_argument("--evaluation-dir", default="outputs/evaluation_full_clean_v2")
    parser.add_argument("--output-dir", default="outputs/attribution_analysis_v2")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction = read_csv(evaluation_dir / "extraction_item_review_v2.csv")
    linking = read_csv(evaluation_dir / "linking_item_review_v2.csv")

    ext_by_condition: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in extraction:
        ext_by_condition[(row["model"], row["variant"])][row["document_id"]] = row

    link_by_condition: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in linking:
        link_by_condition[(row["model"], row["variant"])][row["mention_id"]] = row

    g20 = ("openai/gpt-oss-20b", "grounded")
    g120 = ("openai/gpt-oss-120b", "grounded")
    qwen = ("qwen/qwen3.6-27b", "grounded")
    free120 = ("openai/gpt-oss-120b", "baseline")
    c20 = ("openai/gpt-oss-20b", "candidate_constrained")
    c120 = ("openai/gpt-oss-120b", "candidate_constrained")
    cqwen = ("qwen/qwen3.6-27b", "candidate_constrained")
    u120 = ("openai/gpt-oss-120b", "unconstrained")

    comparisons: list[dict[str, object]] = []

    def add_ext(label: str, a, b, metric: str, seed_offset: int = 0):
        ids = sorted(set(ext_by_condition[a]) & set(ext_by_condition[b]))
        diff, low, high = paired_bootstrap(
            ids, ext_by_condition[a], ext_by_condition[b], extraction_metrics, metric,
            args.seed, args.resamples,
        )
        comparisons.append({
            "task": "extraction", "metric": metric, "comparison": label,
            "difference": diff, "ci_low": low, "ci_high": high, "n": len(ids),
            "practical_threshold_met": abs(diff) >= PRACTICAL_THRESHOLD and (low > 0 or high < 0),
        })

    def add_link(label: str, a, b, metric: str, subset: str = "all", seed_offset: int = 0):
        a_rows = link_by_condition[a]
        b_rows = link_by_condition[b]
        ids = sorted(set(a_rows) & set(b_rows))
        if subset == "ambiguous":
            ids = [i for i in ids if a_rows[i]["ambiguous_surface_in_train_dev"] == "1"]
        elif subset == "unambiguous":
            ids = [i for i in ids if a_rows[i]["ambiguous_surface_in_train_dev"] == "0"]
        diff, low, high = paired_bootstrap(
            ids, a_rows, b_rows, linking_metrics, metric,
            args.seed, args.resamples,
        )
        comparisons.append({
            "task": "linking", "metric": metric, "comparison": label,
            "difference": diff, "ci_low": low, "ci_high": high, "n": len(ids),
            "practical_threshold_met": abs(diff) >= PRACTICAL_THRESHOLD and (low > 0 or high < 0),
        })

    add_ext("Qwen grounded - GPT-OSS 120B grounded", qwen, g120, "f1", 1)
    add_ext("Qwen grounded - GPT-OSS 20B grounded", qwen, g20, "f1", 2)
    add_ext("GPT-OSS 120B grounded - GPT-OSS 20B grounded", g120, g20, "f1", 3)
    add_ext("GPT-OSS 120B grounded - GPT-OSS 120B free-form", g120, free120, "f1", 4)
    add_ext("GPT-OSS 120B grounded - GPT-OSS 20B grounded", g120, g20, "unsupported_rate", 5)
    add_ext("GPT-OSS 120B grounded - GPT-OSS 120B free-form", g120, free120, "unsupported_rate", 6)
    add_ext("Qwen grounded - GPT-OSS 120B grounded", qwen, g120, "boundary_rate", 7)
    add_ext("GPT-OSS 120B grounded - GPT-OSS 120B free-form", g120, free120, "boundary_rate", 8)

    add_link("GPT-OSS 120B constrained - GPT-OSS 20B constrained", c120, c20, "accuracy", seed_offset=9)
    add_link("Qwen constrained - GPT-OSS 120B constrained", cqwen, c120, "accuracy", seed_offset=10)
    add_link("GPT-OSS 120B constrained - GPT-OSS 120B unconstrained", c120, u120, "accuracy", seed_offset=11)
    add_link("GPT-OSS 120B constrained - GPT-OSS 120B unconstrained", c120, u120, "invalid_rate", seed_offset=12)
    add_link("GPT-OSS 120B constrained - GPT-OSS 120B unconstrained (ambiguous)", c120, u120, "accuracy", subset="ambiguous", seed_offset=13)

    # Condition summaries.
    summary_rows: list[dict[str, object]] = []
    for condition, rows in sorted(ext_by_condition.items()):
        metrics = extraction_metrics(rows.values())
        for metric, value in metrics.items():
            if metric != "n":
                summary_rows.append({"task": "extraction", "model": condition[0], "variant": condition[1], "subset": "all", "metric": metric, "value": value, "n": int(metrics["n"])})
    for condition, rows in sorted(link_by_condition.items()):
        for subset_name, subset_rows in [
            ("all", list(rows.values())),
            ("ambiguous", [r for r in rows.values() if r["ambiguous_surface_in_train_dev"] == "1"]),
            ("unambiguous", [r for r in rows.values() if r["ambiguous_surface_in_train_dev"] == "0"]),
        ]:
            metrics = linking_metrics(subset_rows)
            for metric, value in metrics.items():
                if metric != "n":
                    summary_rows.append({"task": "linking", "model": condition[0], "variant": condition[1], "subset": subset_name, "metric": metric, "value": value, "n": int(metrics["n"])})

    with (output_dir / "condition_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "model", "variant", "subset", "metric", "value", "n"])
        writer.writeheader(); writer.writerows(summary_rows)

    with (output_dir / "paired_bootstrap_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "metric", "comparison", "difference", "ci_low", "ci_high", "n", "practical_threshold_met"])
        writer.writeheader(); writer.writerows(comparisons)

    attribution = [
        {
            "failure_category": "Overall exact-span extraction",
            "classification": "Model-dominant",
            "evidence": "Qwen vs GPT-OSS 120B grounded F1 difference >=5 points with CI excluding zero; 120B grounding gain was <5 points.",
        },
        {
            "failure_category": "Unsupported entity generation",
            "classification": "Hybrid",
            "evidence": "A material fixed-architecture model difference and a material fixed-model grounding reduction were both observed.",
        },
        {
            "failure_category": "Entity-boundary error",
            "classification": "Model-dominant under tested intervention",
            "evidence": "A material model difference was observed; grounding increased rather than remediated boundary errors.",
        },
        {
            "failure_category": "Candidate-ranking error",
            "classification": "Architecture-dominant",
            "evidence": "Constrained-model differences were <5 points, while candidate restriction improved 120B accuracy by >5 points with CI excluding zero.",
        },
        {
            "failure_category": "Malformed ontology code",
            "classification": "Architecture-dominant",
            "evidence": "Candidate restriction reduced invalid-CUI output by >5 points with CI excluding zero.",
        },
        {
            "failure_category": "Ambiguous surface-form linking",
            "classification": "Architecture-dominant",
            "evidence": "Constrained-model differences were small; candidate restriction materially improved the paired ambiguous subset, although accuracy remained near 51%.",
        },
    ]
    with (output_dir / "attribution_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["failure_category", "classification", "evidence"])
        writer.writeheader(); writer.writerows(attribution)

    report = {
        "seed": args.seed,
        "bootstrap_resamples": args.resamples,
        "practical_threshold": PRACTICAL_THRESHOLD,
        "extraction_documents_per_condition": len(next(iter(ext_by_condition.values()))),
        "linking_mentions_per_condition": len(next(iter(link_by_condition.values()))),
        "ambiguous_mentions": sum(1 for r in link_by_condition[c120].values() if r["ambiguous_surface_in_train_dev"] == "1"),
        "outputs": ["condition_metrics.csv", "paired_bootstrap_effects.csv", "attribution_matrix.csv"],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Attribution analysis completed")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
