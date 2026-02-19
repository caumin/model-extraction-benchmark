# Evidence Log

- item_path: `paper.title`
  - pdf: `inversenet.pdf`
  - page: `2439`
  - ref: `Title`
  - quote: `INVERSENET: Augmenting Model Extraction Attacks with Training Data Inversion`
  - interpretation: `paper_meta/extracted_spec title`

- item_path: `attack.oracle_access.rounding`
  - pdf: `inversenet.pdf`
  - page: `2440`
  - ref: `Section 2 Threat Model`
  - quote: `API only returns the class label (top-1 results) but not the confidence scores`
  - interpretation: `hard-label only setting`

- item_path: `attack.hyperparameters.phase_ratios`
  - pdf: `inversenet.pdf`
  - page: `2443`
  - ref: `Impact of query budget`
  - quote: `The ratio between K1, K2, and K3 is fixed at 0.45:0.45:0.1`
  - interpretation: `phase_ratios=[0.45,0.45,0.1]`

- item_path: `attack.hyperparameters.high_confidence_selection.xi`
  - pdf: `inversenet.pdf`
  - page: `2441`
  - ref: `Section 3.2`
  - quote: `In our experiments, ξ was set to 0.02`
  - interpretation: `hcss_xi=0.02`

- item_path: `datasets[0].splits`
  - pdf: `inversenet.pdf`
  - page: `2445`
  - ref: `Appendix A Dataset of Victim Model`
  - quote: `MNIST ... 60,000 samples as the training set, and 10,000 samples as the test set`
  - interpretation: `MNIST split metadata`

- item_path: `datasets[0].notes`
  - pdf: `inversenet.pdf`
  - page: `2445`
  - ref: `Appendix B Query Dataset`
  - quote: `For the MNIST victim model, we use EMNIST Letters`
  - interpretation: `MNIST surrogate/query dataset = EMNIST letters`

- item_path: `reported_results.tables[0].metrics.mnist_agreement_at_10k`
  - pdf: `inversenet.pdf`
  - page: `2443`
  - ref: `Impact of query budget paragraph`
  - quote: `At a query budget of 10k, the MNIST substitute model reaches an agreement of 93.2%`
  - interpretation: `paper target agreement=0.932 at 10k`

- item_path: `reported_results.tables[0].metrics.cifar10_agreement_at_10k`
  - pdf: `inversenet.pdf`
  - page: `2443`
  - ref: `Impact of query budget paragraph`
  - quote: `the more complex CIFAR10 model yields an agreement of 75.4%`
  - interpretation: `reference point for CIFAR10 agreement`
