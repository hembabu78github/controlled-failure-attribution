# Prompt Templates

The executable source of truth is `scripts/03_run_groq_pilot_v2.py`.  
This file presents the same prompt logic in a compact, human-readable form.

## Shared structured-output requirement

### Extraction

The runner instructs the model to perform biomedical named-entity extraction for a MedMentions-style UMLS benchmark and return exactly one JSON object:

```json
{"entities": [{"text": "..."}]}
```

No prose is allowed. The root object must contain exactly the `entities` key; each entity object contains exactly the `text` key.

### Linking

The runner instructs the model to perform biomedical entity linking to UMLS and return exactly one JSON object:

```json
{"selected_cui": "C1234567"}
```

No prose is allowed.

## Grounded extraction condition

The model is instructed to:

- copy every predicted mention **exactly** as it appears in the input text;
- not normalize, expand, paraphrase, infer, or add concepts not literally present;
- return an empty `entities` array if no biomedical concept is present.

The local evaluator/validator then attempts to locate each returned string in the source text. Strings absent from the source are **retained as unsupported predictions for evaluation**, not silently converted into correct outputs.

The input document is appended after an `INPUT_TEXT` delimiter.

## Free-form extraction baseline

The model is instructed to list biomedical entities identified in the input and may use the wording it considers most appropriate. It returns an empty array if it identifies no biomedical entity.

## Candidate-constrained linking condition

For each mention, the prompt contains:

- the marked mention;
- semantic type;
- local context;
- exactly five candidate CUI/label pairs.

The model is instructed to select **exactly one CUI from the supplied candidate list** and not invent or alter a CUI outside that list.

## Unconstrained linking baseline

The prompt contains the mention, semantic type, and context but no candidate list. The model is instructed to return its single best UMLS CUI from memory even when uncertain.

## Hosted inference controls

The completed run used:

- one user message;
- no system prompt;
- temperature `0`;
- JSON object mode;
- maximum completion tokens: `4096` extraction, `512` linking;
- GPT-OSS: `reasoning_effort="low"`, `include_reasoning=false`;
- Qwen3.6-27B: `reasoning_effort="none"`, `reasoning_format="hidden"`;
- top-p, frequency penalty, and presence penalty left at provider defaults.

For exact string construction, delimiters, field ordering, and request-body behavior, consult `scripts/03_run_groq_pilot_v2.py`.
