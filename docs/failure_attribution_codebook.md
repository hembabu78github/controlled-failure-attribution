# Pilot Failure-Attribution Codebook

This pilot does not claim to empirically validate every category in the full literature-derived taxonomy. It directly measures the following observable categories.

## 1. Unsupported entity generation

A predicted entity is not an exact substring of the input text, or its supplied offsets do not reproduce the predicted text.

Primary evidence:
- Free-form extraction versus grounded extraction
- Unsupported-entity rate

Initial attribution logic:
- Large reduction after grounding with the same model supports an architecture-dominant interpretation.
- Large differences between models under the same grounded pipeline support a model-dominant component.

## 2. Entity boundary error

A predicted entity overlaps a gold mention but does not match its exact start and end offsets.

Primary evidence:
- Exact-span F1
- Boundary-error rate
- Cross-model persistence under grounded extraction

## 3. Candidate-ranking error

The correct CUI appears in the supplied candidate list, but the model selects a different candidate.

Primary evidence:
- Candidate-constrained top-1 CUI accuracy
- Cross-model variation under the same five-candidate architecture

## 4. Unsupported ontology code

The model produces a missing, malformed, or non-candidate CUI.

Primary evidence:
- Unconstrained CUI generation versus candidate-constrained selection
- Invalid-candidate rate

## 5. Ambiguous surface-form error

A normalized surface form maps to more than one CUI in the train/dev candidate index and the selected CUI is wrong in context.

Primary evidence:
- Error rate among `ambiguous_surface_in_train_dev = true`
- Model comparison under the same candidate list

## 6. Unresolved

Use this label where the pilot evidence cannot separate model and architecture effects, or where too few examples are observed.

## Attribution decision rules

- **Model-dominant:** a fixed-architecture model difference reaches the practical threshold with a paired 95% confidence interval excluding zero, while the tested architecture intervention does not produce a qualifying improvement.
- **Architecture-dominant:** the fixed-model architecture intervention produces a qualifying improvement, while fixed-architecture model differences do not.
- **Hybrid:** both model substitution and architecture intervention produce qualifying improvement.
- **Unresolved:** neither criterion is met, the evidence conflicts, or the sample is insufficient.
- **Adverse architecture effect:** a significant worsening is reported separately and never counted as an architecture gain. If the model criterion qualifies and the tested architecture effect is adverse, the category is reported as model-dominant under the tested intervention.

The primary threshold was frozen at five percentage points after the smoke test and before the full run. This is a pragmatic engineering convention rather than a clinically validated minimum. Reviewer-requested sensitivity analyses at three and ten points are included in `results/threshold_sensitivity.csv`.
