# Data Access

MedMentions is an externally maintained benchmark and is not included in this archive.

- Paper: S. Mohan and D. Li, "MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts," AKBC 2019, doi: 10.24432/C5G59C.
- Official repository: https://github.com/chanzuckerberg/MedMentions
- Corpus used: ST21pv with the official train/dev/test split.

The preparation script uses train and development annotations to build the candidate index and reserves the official test annotations for evaluation. Each controlled five-candidate linking question includes the gold CUI by design; this isolates ranking conditional on candidate recall and must be reported transparently.
