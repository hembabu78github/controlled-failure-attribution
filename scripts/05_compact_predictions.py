from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Line {line_number} is not a JSON object."
                )
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compact resumable predictions. For each request key, prefer the "
            "latest successful record; otherwise keep the latest failed record."
        )
    )
    parser.add_argument(
        "--input",
        default="outputs/predictions/full_predictions_v2.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions/full_predictions_v2_clean.jsonl",
    )
    parser.add_argument(
        "--unresolved",
        default="outputs/predictions/unresolved_requests_v2.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    unresolved_path = Path(args.unresolved)

    rows = load_jsonl(input_path)

    first_seen: OrderedDict[str, None] = OrderedDict()
    latest_any: dict[str, dict[str, Any]] = {}
    latest_ok: dict[str, dict[str, Any]] = {}
    missing_key_rows = 0

    for row in rows:
        request_key = str(row.get("request_key", "")).strip()
        if not request_key:
            missing_key_rows += 1
            continue

        if request_key not in first_seen:
            first_seen[request_key] = None

        latest_any[request_key] = row
        if row.get("status") == "ok":
            latest_ok[request_key] = row

    selected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for request_key in first_seen:
        if request_key in latest_ok:
            selected.append(latest_ok[request_key])
        else:
            row = latest_any[request_key]
            selected.append(row)
            unresolved.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    unresolved_path.parent.mkdir(parents=True, exist_ok=True)
    with unresolved_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "request_key",
                "model",
                "task",
                "variant",
                "status",
                "error",
            ]
        )
        for row in unresolved:
            writer.writerow(
                [
                    row.get("request_key", ""),
                    row.get("model", ""),
                    row.get("task", ""),
                    row.get("variant", ""),
                    row.get("status", ""),
                    row.get("error", ""),
                ]
            )

    summary = {
        "input_rows": len(rows),
        "rows_without_request_key": missing_key_rows,
        "unique_request_keys": len(first_seen),
        "successful_unique_requests": len(latest_ok),
        "unresolved_unique_requests": len(unresolved),
        "clean_output": str(output_path),
        "unresolved_output": str(unresolved_path),
    }

    summary_path = output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nCompaction completed")
    print("--------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return 0 if not unresolved else 1


if __name__ == "__main__":
    raise SystemExit(main())
