# Evidence Log

Use this format for each extracted claim:

- item_path: `extracted_spec.yaml key path`
- pdf: `activethief.pdf`
- page: `N`
- ref: `Section / Table / Figure`
- quote: `<=25 words`
- interpretation: `value used in config/code`

## Seed entries

- item_path: `paper.title`
  - pdf: `activethief.pdf`
  - page: `865`
  - ref: `Title block`
  - quote: `ACTIVETHIEF: Model Extraction Using Active Learning and Unannotated Public Data`
  - interpretation: `paper title`

- item_path: `attack.substitute.training.initial_seed_ratio`
  - pdf: `activethief.pdf`
  - page: `869`
  - ref: `Training regime`
  - quote: `We set aside 20% of the query budget for validation, and use 10% as the initial seed samples.`
  - interpretation: `validation_budget_ratio=0.2, initial_seed_ratio=0.1`
