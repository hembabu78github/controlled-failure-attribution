from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CUI_PATTERN = re.compile(r"^C\d{7}$")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
    return rows


def canonical_cui(value: Any) -> str:
    """
    Normalize only harmless presentation differences.

    Examples:
      UMLS:C0004755 -> C0004755
      umls:c0004755 -> C0004755

    The function does not pad, truncate, or repair malformed identifiers.
    """
    text = str(value or "").strip().upper()
    if text.startswith("UMLS:"):
        text = text[5:].strip()
    return text


def is_valid_cui(value: Any) -> bool:
    return bool(CUI_PATTERN.fullmatch(canonical_cui(value)))


def overlap(
    a_start: int, a_end: int, b_start: int, b_end: int
) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def locate_text(
    source: str,
    mention: str,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    start_at = 0
    while True:
        index = source.find(mention, start_at)
        if index < 0:
            return None
        span = (index, index + len(mention))
        if span not in used:
            return span
        start_at = index + 1


def predicted_extraction_spans(
    prediction: dict[str, Any],
    source_text: str,
) -> tuple[list[tuple[int, int, str]], list[str]]:
    parsed = prediction.get("parsed") or {}
    entities = (
        parsed.get("entities", [])
        if isinstance(parsed, dict)
        else []
    )

    located: list[tuple[int, int, str]] = []
    unsupported: list[str] = []
    used: set[tuple[int, int]] = set()

    if not isinstance(entities, list):
        return located, ["<entities field was not an array>"]

    for item in entities:
        if not isinstance(item, dict):
            unsupported.append("<non-object entity>")
            continue

        mention = str(item.get("text", ""))
        if not mention:
            unsupported.append("<empty entity text>")
            continue

        position = locate_text(source_text, mention, used)
        if position is None:
            unsupported.append(mention)
            continue

        start, end = position
        located.append((start, end, mention))
        used.add((start, end))

    return located, unsupported


def prf(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> tuple[float, float, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def write_csv(
    path: Path,
    header: list[str],
    rows: list[list[Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="pilot_data")
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/evaluation_v2",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction_gold = {
        row["document_id"]: row
        for row in load_jsonl(
            input_dir / "extraction_sample.jsonl"
        )
    }
    linking_gold = {
        row["mention_id"]: row
        for row in load_jsonl(
            input_dir / "linking_sample.jsonl"
        )
    }
    predictions = load_jsonl(predictions_path)

    extraction_acc: dict[
        tuple[str, str], dict[str, int]
    ] = defaultdict(lambda: defaultdict(int))
    linking_acc: dict[
        tuple[str, str], dict[str, int]
    ] = defaultdict(lambda: defaultdict(int))

    extraction_review: list[list[Any]] = []
    linking_review: list[list[Any]] = []
    failed_request_rows: list[list[Any]] = []

    for prediction in predictions:
        status = prediction.get("status")
        if status != "ok":
            failed_request_rows.append(
                [
                    prediction.get("request_key", ""),
                    prediction.get("model", ""),
                    prediction.get("task", ""),
                    prediction.get("variant", ""),
                    prediction.get("error", ""),
                ]
            )
            continue

        model = str(prediction["model"])
        variant = str(prediction["variant"])
        condition = (model, variant)

        if prediction["task"] == "extraction":
            gold = extraction_gold.get(
                str(prediction["item_id"])
            )
            if gold is None:
                failed_request_rows.append(
                    [
                        prediction.get("request_key", ""),
                        model,
                        "extraction",
                        variant,
                        "Gold document not found.",
                    ]
                )
                continue

            gold_spans = {
                (int(item["start"]), int(item["end"]))
                for item in gold["gold_mentions"]
            }
            predicted_items, unsupported_items = (
                predicted_extraction_spans(
                    prediction=prediction,
                    source_text=str(gold["text"]),
                )
            )
            predicted_spans = {
                (start, end)
                for start, end, _ in predicted_items
            }

            exact_matches = gold_spans & predicted_spans
            false_positive_spans = (
                predicted_spans - gold_spans
            )
            false_negative_spans = (
                gold_spans - predicted_spans
            )

            boundary_errors = {
                predicted_span
                for predicted_span in false_positive_spans
                if any(
                    overlap(
                        predicted_span[0],
                        predicted_span[1],
                        gold_span[0],
                        gold_span[1],
                    )
                    for gold_span in gold_spans
                )
            }

            acc = extraction_acc[condition]
            acc["tp"] += len(exact_matches)
            acc["fp"] += (
                len(false_positive_spans)
                + len(unsupported_items)
            )
            acc["fn"] += len(false_negative_spans)
            acc["unsupported"] += len(unsupported_items)
            acc["boundary"] += len(boundary_errors)
            acc["predicted"] += (
                len(predicted_spans)
                + len(unsupported_items)
            )
            acc["documents"] += 1

            extraction_review.append(
                [
                    model,
                    variant,
                    prediction["item_id"],
                    len(gold_spans),
                    len(predicted_spans),
                    len(exact_matches),
                    len(false_positive_spans),
                    len(false_negative_spans),
                    len(boundary_errors),
                    len(unsupported_items),
                    " | ".join(unsupported_items[:10]),
                ]
            )

        elif prediction["task"] == "linking":
            gold = linking_gold.get(
                str(prediction["item_id"])
            )
            if gold is None:
                failed_request_rows.append(
                    [
                        prediction.get("request_key", ""),
                        model,
                        "linking",
                        variant,
                        "Gold linking item not found.",
                    ]
                )
                continue

            parsed = prediction.get("parsed") or {}
            raw_selected = (
                parsed.get("selected_cui", "")
                if isinstance(parsed, dict)
                else ""
            )
            selected_cui = canonical_cui(raw_selected)
            gold_cui = canonical_cui(gold["gold_cui"])
            candidate_cuis = {
                canonical_cui(item["cui"])
                for item in gold["candidates"]
            }

            valid_format = is_valid_cui(raw_selected)
            correct = selected_cui == gold_cui
            outside_candidate_list = (
                variant == "candidate_constrained"
                and selected_cui not in candidate_cuis
            )

            acc = linking_acc[condition]
            acc["items"] += 1
            acc["correct"] += int(correct)
            acc["invalid_format"] += int(not valid_format)
            acc["outside_candidate_list"] += int(
                outside_candidate_list
            )
            acc["empty"] += int(not selected_cui)

            linking_review.append(
                [
                    model,
                    variant,
                    prediction["item_id"],
                    gold["mention_text"],
                    gold["semantic_type"],
                    str(gold["gold_cui"]),
                    str(raw_selected),
                    gold_cui,
                    selected_cui,
                    int(valid_format),
                    int(correct),
                    int(outside_candidate_list),
                    int(
                        gold.get(
                            "ambiguous_surface_in_train_dev",
                            False,
                        )
                    ),
                ]
            )

    metric_rows: list[list[Any]] = []
    condition_summary: dict[str, Any] = {
        "prediction_file": str(predictions_path),
        "total_records": len(predictions),
        "failed_or_unmatched_records": len(
            failed_request_rows
        ),
        "conditions": {},
    }

    for (
        model,
        variant,
    ), acc in sorted(extraction_acc.items()):
        precision, recall, f1 = prf(
            acc["tp"], acc["fp"], acc["fn"]
        )
        unsupported_rate = safe_rate(
            acc["unsupported"], acc["predicted"]
        )
        boundary_rate = safe_rate(
            acc["boundary"], acc["predicted"]
        )

        metrics = {
            "documents": acc["documents"],
            "exact_span_precision": precision,
            "exact_span_recall": recall,
            "exact_span_f1": f1,
            "unsupported_entity_rate": unsupported_rate,
            "boundary_error_rate": boundary_rate,
            "true_positive_spans": acc["tp"],
            "false_positive_spans": acc["fp"],
            "false_negative_spans": acc["fn"],
        }
        condition_summary["conditions"][
            f"extraction|{model}|{variant}"
        ] = metrics

        for metric, value in metrics.items():
            if metric == "documents":
                continue
            metric_rows.append(
                [
                    "extraction",
                    model,
                    variant,
                    metric,
                    value,
                    acc["documents"],
                ]
            )

    for (
        model,
        variant,
    ), acc in sorted(linking_acc.items()):
        accuracy = safe_rate(
            acc["correct"], acc["items"]
        )
        invalid_format_rate = safe_rate(
            acc["invalid_format"], acc["items"]
        )
        outside_rate = safe_rate(
            acc["outside_candidate_list"],
            acc["items"],
        )
        empty_rate = safe_rate(
            acc["empty"], acc["items"]
        )

        metrics = {
            "items": acc["items"],
            "top1_cui_accuracy": accuracy,
            "invalid_cui_format_rate": invalid_format_rate,
            "outside_candidate_list_rate": outside_rate,
            "empty_cui_rate": empty_rate,
        }
        condition_summary["conditions"][
            f"linking|{model}|{variant}"
        ] = metrics

        for metric, value in metrics.items():
            if metric == "items":
                continue
            metric_rows.append(
                [
                    "linking",
                    model,
                    variant,
                    metric,
                    value,
                    acc["items"],
                ]
            )

    metrics_path = output_dir / "pilot_metrics_v2.csv"
    write_csv(
        metrics_path,
        ["task", "model", "variant", "metric", "value", "n"],
        metric_rows,
    )

    extraction_review_path = (
        output_dir / "extraction_item_review_v2.csv"
    )
    write_csv(
        extraction_review_path,
        [
            "model",
            "variant",
            "document_id",
            "gold_spans",
            "predicted_spans",
            "exact_matches",
            "false_positive_spans",
            "false_negative_spans",
            "boundary_errors",
            "unsupported_entities",
            "unsupported_examples",
        ],
        extraction_review,
    )

    linking_review_path = (
        output_dir / "linking_item_review_v2.csv"
    )
    write_csv(
        linking_review_path,
        [
            "model",
            "variant",
            "mention_id",
            "mention_text",
            "semantic_type",
            "gold_cui_raw",
            "predicted_cui_raw",
            "gold_cui_canonical",
            "predicted_cui_canonical",
            "valid_cui_format",
            "correct",
            "outside_candidate_list",
            "ambiguous_surface_in_train_dev",
        ],
        linking_review,
    )

    failed_path = (
        output_dir / "failed_or_unmatched_requests_v2.csv"
    )
    write_csv(
        failed_path,
        [
            "request_key",
            "model",
            "task",
            "variant",
            "error",
        ],
        failed_request_rows,
    )

    summary_path = output_dir / "evaluation_summary_v2.json"
    summary_path.write_text(
        json.dumps(
            condition_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nEvaluation completed")
    print("--------------------")
    print(
        f"Prediction records: {len(predictions)}"
    )
    print(
        "Failed or unmatched records: "
        f"{len(failed_request_rows)}"
    )
    print(f"Metrics:          {metrics_path.resolve()}")
    print(
        "Extraction review: "
        f"{extraction_review_path.resolve()}"
    )
    print(
        "Linking review:    "
        f"{linking_review_path.resolve()}"
    )
    print(f"Summary:          {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
