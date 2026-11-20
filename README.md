# Controlled Failure Attribution for Trustworthy Clinical Information Extraction

Code and derived results for the OAIC'26 paper:

**Controlled Failure Attribution for Trustworthy Clinical Information Extraction: Separating Model and Pipeline Effects**

**Authors:** Hemachandran Babu and Priyadarsini K

## What this repository contains

This repository contains only the files needed to understand and reproduce the reported experiment:

- `scripts/` — deterministic sample preparation, hosted inference, evaluation, compaction, bootstrap attribution, and reviewer-requested sensitivity analysis.
- `protocol_config.json` — frozen study configuration.
- `PROMPTS.md` — reviewer-friendly prompt documentation; the executable source of truth is `scripts/03_run_groq_pilot_v2.py`.
- `docs/failure_attribution_codebook.md` — operational definitions and attribution rules.
- `results/` — compact derived outputs that support the paper's tables and claims.
- `DATA_ACCESS.md` — instructions for obtaining MedMentions separately.
- `CITATION.cff` — citation metadata.
- `requirements.txt` — Python dependencies.

The raw MedMentions corpus and full hosted-model response logs are **not redistributed**.

## Study design

Two intervention axes were evaluated:

1. **Model substitution under a fixed pipeline**
   - `openai/gpt-oss-20b`
   - `openai/gpt-oss-120b`
   - `qwen/qwen3.6-27b`

2. **Pipeline intervention under a fixed model (`openai/gpt-oss-120b`)**
   - free-form vs. source-grounded extraction
   - unconstrained UMLS generation vs. five-candidate CUI selection

Frozen protocol:

- MedMentions ST21pv official splits
- deterministic seed: `20260721`
- extraction sample: 100 test documents
- linking sample: 300 mentions
- ambiguous linking subset: 98 mentions
- candidate set size: 5
- bootstrap resamples: 10,000
- primary practical-effect threshold: 5 percentage points
- sensitivity thresholds: 3 and 10 percentage points
- total unique model-condition requests: 1,600

See `protocol_config.json` and `results/run_completion_summary.json`.

## Environment

Python 3.11 was used.

```bash
python -m pip install -r requirements.txt
```

For hosted inference, set your Groq key outside the code:

```powershell
$env:GROQ_API_KEY="your-key"
```

## Data

Download MedMentions from its official source and place the required files under:

```text
data/medmentions/
```

See `DATA_ACCESS.md`.

## Reproduction sequence

From the repository root:

```powershell
py -3.11 scripts\02_prepare_pilot_sample_optimized.py
py -3.11 scripts\03_run_groq_pilot_v2.py
py -3.11 scripts\05_compact_predictions.py
py -3.11 scripts\04_evaluate_pilot_v2.py `
  --predictions outputs\predictions\full_predictions_v2_clean.jsonl `
  --output-dir outputs\evaluation_full_clean_v2
py -3.11 scripts\06_analyze_attribution.py `
  --evaluation-dir outputs\evaluation_full_clean_v2 `
  --output-dir outputs\attribution_analysis_v2
py -3.11 scripts\07_reviewer_sensitivity_and_subset_ci.py
```

The inference runner is resumable. The compaction step keeps the successful record for each request key after quota/network retries.

## Exact prompts and inference controls

See `PROMPTS.md` and `scripts/03_run_groq_pilot_v2.py`.

The completed experiment used:

- one user message;
- no system prompt;
- temperature `0`;
- JSON object mode;
- maximum completion tokens: 4,096 extraction / 512 linking;
- GPT-OSS: `reasoning_effort="low"` with returned reasoning suppressed;
- Qwen3.6-27B: `reasoning_effort="none"` with hidden reasoning;
- top-p and penalty parameters left at provider defaults.

The provider exposed different minimum reasoning controls across model families, so this remains a disclosed inference-configuration limitation.

## Results included

- `condition_metrics.csv` — aggregate extraction and linking metrics, including ambiguous/unambiguous subsets.
- `paired_bootstrap_effects.csv` — controlled model and pipeline effects with 95% bootstrap confidence intervals.
- `attribution_matrix.csv` — primary failure-attribution verdicts.
- `threshold_sensitivity.csv` — reviewer-requested 3/5/10-point sensitivity analysis.
- `linking_subset_accuracy_cis.csv` — ambiguous/unambiguous linking confidence intervals.
- `run_completion_summary.json` — compact completion and zero-failure audit.

## Important interpretation limits

- MedMentions contains biomedical abstracts rather than longitudinal clinical notes.
- Five-candidate linking is a **controlled reranking** experiment: the gold CUI is deliberately present, so accuracy is conditional on candidate recall.
- Only source grounding and candidate restriction were tested as pipeline interventions.
- One deterministic seed selected the extraction sample.
- Immutable hosted-model snapshot identifiers were unavailable.
- Attribution verdicts are conditional on the tested models, interventions, failure categories, and practical-effect threshold.

## License

No license is included in this release candidate. Before making the repository public, add the license that the authors intend to grant.
