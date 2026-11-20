from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class OutputValidationError(ValueError):
    pass


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for row in load_jsonl(path):
        if row.get("status") == "ok" and row.get("request_key"):
            keys.add(str(row["request_key"]))
    return keys


def parse_json_object(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise OutputValidationError("Model returned empty final content.")

    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(
            f"Final content is not valid JSON: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise OutputValidationError("Final JSON must be an object.")
    return value


def validate_output(
    parsed: dict[str, Any],
    task: str,
    variant: str,
) -> None:
    if task == "extraction":
        entities = parsed.get("entities")
        if not isinstance(entities, list):
            raise OutputValidationError(
                "Extraction JSON must contain an 'entities' array."
            )
        for index, item in enumerate(entities):
            if not isinstance(item, dict):
                raise OutputValidationError(
                    f"entities[{index}] must be an object."
                )
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise OutputValidationError(
                    f"entities[{index}].text must be a non-empty string."
                )
        return

    if task == "linking":
        selected = parsed.get("selected_cui")
        if not isinstance(selected, str) or not selected.strip():
            raise OutputValidationError(
                "Linking JSON must contain a non-empty 'selected_cui' string."
            )
        return

    raise OutputValidationError(f"Unsupported task: {task}")


def model_controls(model: str) -> dict[str, Any]:
    """
    Keep the visible response free of reasoning text.

    GPT-OSS:
      - reasoning_format is not supported
      - use include_reasoning=false and low reasoning effort

    Qwen 3.6:
      - disable reasoning
      - hide any reasoning channel
    """
    if model.startswith("openai/gpt-oss-"):
        return {
            "include_reasoning": False,
            "reasoning_effort": "low",
        }

    if model == "qwen/qwen3.6-27b":
        return {
            "reasoning_effort": "none",
            "reasoning_format": "hidden",
        }

    return {}


def groq_chat(
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "response_format": {"type": "json_object"},
    }
    body.update(model_controls(model))

    encoded = json.dumps(body).encode("utf-8")
    last_error = ""

    for attempt in range(1, max_retries + 1):
        request = urllib.request.Request(
            API_URL,
            data=encoded,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "OAIC-Paper4-Pilot/2.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                payload = json.loads(response.read().decode("utf-8"))

            choice = payload["choices"][0]
            message = choice.get("message", {})
            content = message.get("content") or ""

            if not content.strip():
                reasoning = message.get("reasoning") or ""
                finish_reason = choice.get("finish_reason")
                usage = payload.get("usage", {})
                raise OutputValidationError(
                    "Empty final content. "
                    f"finish_reason={finish_reason!r}; "
                    f"reasoning_chars={len(reasoning)}; "
                    f"usage={usage}"
                )

            return content, payload

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                break

        except OutputValidationError:
            raise

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            time.sleep(min(30, 2 ** attempt))

    raise RuntimeError(last_error)


def extraction_prompt(row: dict[str, Any], variant: str) -> str:
    text = str(row["text"])

    common = (
        "Perform biomedical named-entity extraction for a MedMentions-style "
        "UMLS concept benchmark. Return one valid JSON object and no prose. "
        "The root object must contain exactly one key named entities. "
        "entities must be an array of objects, and every object must contain "
        "exactly one key named text."
    )

    if variant == "grounded":
        instructions = (
            "Copy every predicted mention exactly as it appears in INPUT_TEXT. "
            "Do not normalize, expand, paraphrase, infer, or add a concept that "
            "is not literally present. The local validation layer will assign "
            "character offsets and reject strings absent from the input. "
            "Return an empty entities array when no biomedical concept is present."
        )
    else:
        instructions = (
            "List the biomedical entities you identify in INPUT_TEXT. "
            "Use the entity wording you consider most appropriate. "
            "Return an empty entities array when no biomedical entity is identified."
        )

    return (
        f"{common}\n\n{instructions}\n\n"
        "INPUT_TEXT begins after the delimiter.\n"
        "----- INPUT_TEXT -----\n"
        f"{text}\n"
        "----- END INPUT_TEXT -----"
    )


def linking_prompt(row: dict[str, Any], variant: str) -> str:
    mention = str(row["mention_text"])
    semantic_type = str(row["semantic_type"])
    context = str(row["context"])

    common = (
        "Perform biomedical entity linking to UMLS. Return one valid JSON "
        "object and no prose. The root object must contain exactly one key "
        "named selected_cui. Its value must be a string."
    )

    if variant == "candidate_constrained":
        candidate_lines = "\n".join(
            f"{item['cui']} | {item['label']}"
            for item in row["candidates"]
        )
        instructions = (
            "Choose exactly one CUI from CANDIDATES. "
            "Do not invent, alter, or return any CUI outside that list."
        )
        candidate_block = (
            "\n\nCANDIDATES:\n"
            f"{candidate_lines}"
        )
    else:
        instructions = (
            "Return the UMLS CUI that best matches the marked mention in context. "
            "When uncertain, still return your single best CUI."
        )
        candidate_block = ""

    return (
        f"{common}\n\n{instructions}\n\n"
        f"MENTION: {mention}\n"
        f"SEMANTIC_TYPE: {semantic_type}\n"
        f"CONTEXT:\n{context}"
        f"{candidate_block}"
    )


def iter_jobs(
    models: list[str],
    primary: str,
    extraction_rows: list[dict[str, Any]],
    linking_rows: list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    # Fixed architecture, varied model.
    for model in models:
        for row in extraction_rows:
            yield {
                "task": "extraction",
                "variant": "grounded",
                "model": model,
                "item": row,
                "item_id": row["document_id"],
            }
        for row in linking_rows:
            yield {
                "task": "linking",
                "variant": "candidate_constrained",
                "model": model,
                "item": row,
                "item_id": row["mention_id"],
            }

    # Fixed model, varied architecture.
    for row in extraction_rows:
        yield {
            "task": "extraction",
            "variant": "baseline",
            "model": primary,
            "item": row,
            "item_id": row["document_id"],
        }
    for row in linking_rows:
        yield {
            "task": "linking",
            "variant": "unconstrained",
            "model": primary,
            "item": row,
            "item_id": row["mention_id"],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--input-dir", default="pilot_data")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of extraction and linking items per condition. "
             "Use 5 for the smoke test. Zero means the full sample.",
    )
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print(
            'ERROR: Set GROQ_API_KEY first. '
            'PowerShell: $env:GROQ_API_KEY="your-key"',
            file=sys.stderr,
        )
        return 2

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    models = list(config["models"])
    primary = str(config["primary_architecture_model"])

    extraction_rows = load_jsonl(
        Path(args.input_dir) / "extraction_sample.jsonl"
    )
    linking_rows = load_jsonl(
        Path(args.input_dir) / "linking_sample.jsonl"
    )

    if args.limit > 0:
        extraction_rows = extraction_rows[: args.limit]
        linking_rows = linking_rows[: args.limit]

    filename = (
        "smoke_predictions_v2.jsonl"
        if args.limit
        else "full_predictions_v2.jsonl"
    )
    output_path = Path(args.output_dir) / filename
    done = completed_keys(output_path)

    new_requests = 0
    ok_count = 0
    error_count = 0

    for job in iter_jobs(
        models, primary, extraction_rows, linking_rows
    ):
        request_key = (
            f"{job['task']}|{job['variant']}|"
            f"{job['model']}|{job['item_id']}"
        )
        if request_key in done:
            print(f"SKIP  {request_key}")
            continue

        if job["task"] == "extraction":
            prompt = extraction_prompt(job["item"], job["variant"])
            max_tokens = 4096
        else:
            prompt = linking_prompt(job["item"], job["variant"])
            max_tokens = 512

        started = time.time()
        record: dict[str, Any] = {
            "request_key": request_key,
            "task": job["task"],
            "variant": job["variant"],
            "model": job["model"],
            "item_id": job["item_id"],
            "script_version": "2.0",
        }

        try:
            content, raw = groq_chat(
                api_key=api_key,
                model=job["model"],
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=max_tokens,
                max_retries=int(config.get("max_retries", 5)),
            )
            parsed = parse_json_object(content)
            validate_output(
                parsed=parsed,
                task=job["task"],
                variant=job["variant"],
            )

            record["raw_content"] = content
            record["parsed"] = parsed
            record["usage"] = raw.get("usage", {})
            choice = raw.get("choices", [{}])[0]
            record["finish_reason"] = choice.get("finish_reason")
            record["status"] = "ok"
            ok_count += 1

        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            error_count += 1

        record["elapsed_seconds"] = round(
            time.time() - started, 3
        )
        append_jsonl(output_path, record)
        new_requests += 1

        print(
            f"{record['status'].upper():5} "
            f"{job['task']:10} "
            f"{job['variant']:22} "
            f"{job['model']} "
            f"{job['item_id']}"
        )
        time.sleep(
            float(config.get("request_delay_seconds", 1.0))
        )

    print("\nRun summary")
    print("-----------")
    print(f"New requests: {new_requests}")
    print(f"OK:           {ok_count}")
    print(f"Errors:       {error_count}")
    print(f"Output:       {output_path.resolve()}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
