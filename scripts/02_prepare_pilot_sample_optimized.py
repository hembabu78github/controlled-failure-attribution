from __future__ import annotations

import argparse
import csv
import difflib
import gzip
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Annotation:
    start: int
    end: int
    text: str
    semantic_type: str
    cui: str


@dataclass
class Document:
    pmid: str
    title: str = ""
    abstract: str = ""
    annotations: list[Annotation] | None = None

    @property
    def text(self) -> str:
        if self.title and self.abstract:
            return f"{self.title} {self.abstract}"
        return self.title or self.abstract

    def __post_init__(self) -> None:
        if self.annotations is None:
            self.annotations = []


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def tokens(text: str) -> set[str]:
    return {tok for tok in normalize(text).split() if len(tok) >= 2}


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_pubtator(path: Path) -> dict[str, Document]:
    docs: dict[str, Document] = {}
    line_count = 0
    started = time.time()

    print(f"Reading compressed corpus: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            line_count += 1
            if line_count % 250000 == 0:
                print(
                    f"  parsed {line_count:,} lines; "
                    f"{len(docs):,} documents; "
                    f"{time.time() - started:.1f}s"
                )

            line = raw.rstrip("\n")
            if not line:
                continue

            if "|t|" in line:
                pmid, title = line.split("|t|", 1)
                docs.setdefault(pmid, Document(pmid=pmid)).title = title
                continue

            if "|a|" in line:
                pmid, abstract = line.split("|a|", 1)
                docs.setdefault(pmid, Document(pmid=pmid)).abstract = abstract
                continue

            fields = line.split("\t")
            if len(fields) < 6:
                continue

            pmid, start, end, mention, semantic_type, cui = fields[:6]
            docs.setdefault(pmid, Document(pmid=pmid)).annotations.append(
                Annotation(
                    start=int(start),
                    end=int(end),
                    text=mention,
                    semantic_type=semantic_type,
                    cui=cui,
                )
            )

    print(
        f"Corpus parsed: {len(docs):,} documents from {line_count:,} lines "
        f"in {time.time() - started:.1f}s"
    )
    return docs


def read_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def stratified_sample(
    docs: list[Document], count: int, seed: int
) -> list[Document]:
    rng = random.Random(seed)
    buckets: dict[str, list[Document]] = {"low": [], "medium": [], "high": []}

    for doc in docs:
        n = len(doc.annotations or [])
        key = "low" if n <= 5 else "medium" if n <= 15 else "high"
        buckets[key].append(doc)

    target = {
        "low": round(count * 0.30),
        "medium": round(count * 0.40),
        "high": count - round(count * 0.30) - round(count * 0.40),
    }

    selected: list[Document] = []
    leftovers: list[Document] = []

    for key in ("low", "medium", "high"):
        rng.shuffle(buckets[key])
        take = min(target[key], len(buckets[key]))
        selected.extend(buckets[key][:take])
        leftovers.extend(buckets[key][take:])

    if len(selected) < count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])

    return sorted(selected[:count], key=lambda item: item.pmid)


def best_label(surface_counter: Counter[str]) -> str:
    return sorted(
        surface_counter.items(),
        key=lambda pair: (-pair[1], len(pair[0]), pair[0].lower()),
    )[0][0]


def build_candidate_index(
    docs: Iterable[Document],
) -> tuple[
    dict[str, str],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, list[str]],
    dict[str, str],
]:
    cui_surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    cui_types: dict[str, set[str]] = defaultdict(set)
    normalized_to_cuis: dict[str, set[str]] = defaultdict(set)

    print("Building train/dev candidate index...")
    ann_count = 0
    for doc in docs:
        for ann in doc.annotations or []:
            ann_count += 1
            cui_surfaces[ann.cui][ann.text] += 1
            cui_types[ann.cui].add(ann.semantic_type)
            normalized_to_cuis[normalize(ann.text)].add(ann.cui)

    labels = {cui: best_label(counter) for cui, counter in cui_surfaces.items()}
    normalized_labels = {cui: normalize(label) for cui, label in labels.items()}

    token_to_cuis: dict[str, set[str]] = defaultdict(set)
    type_to_cuis: dict[str, list[str]] = defaultdict(list)

    for cui, label in labels.items():
        for token in tokens(label):
            token_to_cuis[token].add(cui)
        for semantic_type in sorted(cui_types.get(cui, set())):
            type_to_cuis[semantic_type].append(cui)

    for semantic_type in type_to_cuis:
        type_to_cuis[semantic_type].sort()

    print(
        f"Candidate index built from {ann_count:,} annotations: "
        f"{len(labels):,} CUIs, {len(token_to_cuis):,} lexical tokens"
    )
    return (
        labels,
        cui_types,
        normalized_to_cuis,
        token_to_cuis,
        type_to_cuis,
        normalized_labels,
    )


def deterministic_subset(items: list[str], limit: int, seed_text: str) -> list[str]:
    if len(items) <= limit:
        return items[:]
    rng = random.Random(stable_int(seed_text))
    indices = sorted(rng.sample(range(len(items)), limit))
    return [items[index] for index in indices]


def make_candidates(
    ann: Annotation,
    labels: dict[str, str],
    cui_types: dict[str, set[str]],
    normalized_to_cuis: dict[str, set[str]],
    token_to_cuis: dict[str, set[str]],
    type_to_cuis: dict[str, list[str]],
    normalized_labels: dict[str, str],
    count: int,
    max_lexical_pool: int = 600,
    max_type_pool: int = 400,
) -> list[dict[str, str]]:
    mention_norm = normalize(ann.text)
    mention_tokens = tokens(ann.text)

    pool: set[str] = set(normalized_to_cuis.get(mention_norm, set()))

    # Lexically related concepts are usually the most useful distractors.
    lexical_candidates: set[str] = set()
    for token in mention_tokens:
        lexical_candidates.update(token_to_cuis.get(token, set()))

    lexical_list = sorted(lexical_candidates)
    if len(lexical_list) > max_lexical_pool:
        lexical_list = deterministic_subset(
            lexical_list,
            max_lexical_pool,
            f"lexical|{ann.text}|{ann.semantic_type}",
        )
    pool.update(lexical_list)

    # Add a bounded sample from the same semantic type so the pool stays plausible.
    same_type = type_to_cuis.get(ann.semantic_type, [])
    pool.update(
        deterministic_subset(
            same_type,
            max_type_pool,
            f"type|{ann.text}|{ann.semantic_type}",
        )
    )

    # Always include the gold CUI required for the controlled ranking task.
    pool.add(ann.cui)

    scored: list[tuple[float, str]] = []
    for cui in pool:
        if cui == ann.cui:
            continue
        label_norm = normalized_labels.get(cui, "")
        similarity = difflib.SequenceMatcher(
            None, mention_norm, label_norm
        ).ratio()
        type_bonus = 0.15 if ann.semantic_type in cui_types.get(cui, set()) else 0.0
        exact_bonus = 0.35 if label_norm == mention_norm else 0.0
        scored.append((similarity + type_bonus + exact_bonus, cui))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [ann.cui] + [cui for _, cui in scored[: max(0, count - 1)]]

    # In the unlikely case the bounded pool is too small, fill deterministically.
    if len(selected) < count:
        fallback = [
            cui
            for cui in deterministic_subset(
                sorted(labels),
                min(500, len(labels)),
                f"fallback|{ann.text}|{ann.cui}",
            )
            if cui not in selected
        ]
        selected.extend(fallback[: count - len(selected)])

    candidates = [
        {"cui": cui, "label": labels.get(cui, ann.text)}
        for cui in selected[:count]
    ]

    # Rotate deterministically so the gold CUI is not always first.
    if candidates:
        shift = stable_int(
            f"{ann.cui}|{ann.start}|{ann.end}"
        ) % len(candidates)
        candidates = candidates[shift:] + candidates[:shift]

    return candidates


def sentence_context(
    text: str, start: int, end: int, window: int = 220
) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    context = (
        text[left:start]
        + "[MENTION]"
        + text[start:end]
        + "[/MENTION]"
        + text[end:right]
    )
    return context.strip()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/medmentions_st21pv")
    parser.add_argument("--output-dir", default="pilot_data")
    parser.add_argument("--documents", type=int, default=100)
    parser.add_argument("--max-linking-items", type=int, default=300)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = data_dir / "corpus_pubtator.txt.gz"
    required = [
        corpus,
        data_dir / "corpus_pubtator_pmids_trng.txt",
        data_dir / "corpus_pubtator_pmids_dev.txt",
        data_dir / "corpus_pubtator_pmids_test.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset files. Run 01_download_medmentions.py first:\n"
            + "\n".join(missing)
        )

    docs = parse_pubtator(corpus)
    train_ids = read_ids(data_dir / "corpus_pubtator_pmids_trng.txt")
    dev_ids = read_ids(data_dir / "corpus_pubtator_pmids_dev.txt")
    test_ids = read_ids(data_dir / "corpus_pubtator_pmids_test.txt")

    index_docs = [
        docs[pmid]
        for pmid in sorted(train_ids | dev_ids)
        if pmid in docs
    ]
    eligible_test = [
        docs[pmid]
        for pmid in sorted(test_ids)
        if pmid in docs and docs[pmid].text and docs[pmid].annotations
    ]

    (
        labels,
        cui_types,
        normalized_to_cuis,
        token_to_cuis,
        type_to_cuis,
        normalized_labels,
    ) = build_candidate_index(index_docs)

    selected_docs = stratified_sample(
        eligible_test, args.documents, args.seed
    )
    print(
        f"Selected {len(selected_docs)} test documents from "
        f"{len(eligible_test):,} eligible documents."
    )

    extraction_rows: list[dict] = []
    all_linking_rows: list[dict] = []

    for doc_number, doc in enumerate(selected_docs, start=1):
        if doc_number == 1 or doc_number % 10 == 0:
            print(
                f"Preparing document {doc_number}/{len(selected_docs)} "
                f"(linking rows so far: {len(all_linking_rows):,})"
            )

        gold_mentions = [
            {
                "start": ann.start,
                "end": ann.end,
                "text": ann.text,
                "semantic_type": ann.semantic_type,
                "cui": ann.cui,
            }
            for ann in sorted(
                doc.annotations or [],
                key=lambda a: (a.start, a.end, a.cui),
            )
        ]
        extraction_rows.append(
            {
                "document_id": doc.pmid,
                "text": doc.text,
                "gold_mentions": gold_mentions,
            }
        )

        for position, ann in enumerate(doc.annotations or []):
            candidates = make_candidates(
                ann=ann,
                labels=labels,
                cui_types=cui_types,
                normalized_to_cuis=normalized_to_cuis,
                token_to_cuis=token_to_cuis,
                type_to_cuis=type_to_cuis,
                normalized_labels=normalized_labels,
                count=args.candidate_count,
            )
            ambiguous = (
                len(normalized_to_cuis.get(normalize(ann.text), set())) > 1
            )
            all_linking_rows.append(
                {
                    "mention_id": (
                        f"{doc.pmid}:{ann.start}:{ann.end}:{position}"
                    ),
                    "document_id": doc.pmid,
                    "mention_text": ann.text,
                    "start": ann.start,
                    "end": ann.end,
                    "semantic_type": ann.semantic_type,
                    "context": sentence_context(
                        doc.text, ann.start, ann.end
                    ),
                    "gold_cui": ann.cui,
                    "candidates": candidates,
                    "ambiguous_surface_in_train_dev": ambiguous,
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(all_linking_rows)
    linking_rows = all_linking_rows[: args.max_linking_items]

    write_jsonl(
        output_dir / "extraction_sample.jsonl",
        extraction_rows,
    )
    write_jsonl(
        output_dir / "linking_sample.jsonl",
        linking_rows,
    )

    with (output_dir / "sample_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["document_id", "characters", "gold_mentions"]
        )
        for row in extraction_rows:
            writer.writerow(
                [
                    row["document_id"],
                    len(row["text"]),
                    len(row["gold_mentions"]),
                ]
            )

    summary = {
        "script_version": "optimized-1.0",
        "seed": args.seed,
        "source_split": "official MedMentions test split",
        "candidate_index_source": "official train + dev splits only",
        "documents": len(extraction_rows),
        "available_linking_items_before_cap": len(all_linking_rows),
        "linking_items": len(linking_rows),
        "candidate_count": args.candidate_count,
        "ambiguous_linking_items": sum(
            1
            for row in linking_rows
            if row["ambiguous_surface_in_train_dev"]
        ),
        "unique_gold_cuis": len(
            {row["gold_cui"] for row in linking_rows}
        ),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nCompleted successfully:")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved pilot files under: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
