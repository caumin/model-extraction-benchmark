# Evidence Log

- item_path: `paper.title`
  - pdf: `disguide.pdf`
  - page: `1`
  - ref: `Title`
  - quote: `DisGUIDE: Disagreement-Guided Data-Free Model Extraction`
  - interpretation: `paper_meta title`

- item_path: `attack.query_budget.cifar10`
  - source: `official_repo_clones/disguide/run_cifar-10.sh`
  - ref: `query_budget=20`
  - quote: `query_budget=20  # Query budget in millions`
  - interpretation: `budget=20,000,000 images`

- item_path: `attack.hyperparameters.generator_lr`
  - source: `official_repo_clones/disguide/run_cifar-10.sh`
  - ref: `--lr-G 1e-4`
  - quote: `--lr-G 1e-4`
  - interpretation: `generator_lr=0.0001`

- item_path: `attack.hyperparameters.ensemble_size`
  - source: `official_repo_clones/disguide/run_cifar-10.sh`
  - ref: `ensemble_size=2`
  - quote: `Value must be 2 or higher for DisGUIDE`
  - interpretation: `enforce ensemble_size >= 2`
