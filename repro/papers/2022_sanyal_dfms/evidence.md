# Evidence Log

- item_path: `paper.title`
  - pdf: `DFMS.pdf`
  - page: `15284`
  - ref: `Title`
  - quote: `Towards Data-Free Model Stealing in a Hard Label Setting`
  - interpretation: `paper_meta/extracted_spec title`

- item_path: `attack.oracle_access.type`
  - pdf: `DFMS.pdf`
  - page: `15284`
  - ref: `Abstract`
  - quote: `most of the APIs allow access to only the top-1 labels`
  - interpretation: `hard-label (top-1) setting`

- item_path: `attack.query_budget.cifar10`
  - pdf: `DFMS.pdf`
  - page: `15288`
  - ref: `Section 3.5`
  - quote: `We set the query limit NQ to 8 million ... on CIFAR-10`
  - interpretation: `cifar10 budget=8,000,000`

- item_path: `attack.hyperparameters.init_clone_queries_nc`
  - pdf: `DFMS.pdf`
  - page: `15288`
  - ref: `Section 3.5`
  - quote: `For our experiments, we set nC as 50,000`
  - interpretation: `init clone query constant=50,000`

- item_path: `attack.hyperparameters.generator_optimizer`
  - pdf: `DFMS.pdf`
  - page: `15289`
  - ref: `Section 4.1`
  - quote: `GAN is trained with an Adam optimizer ... 2×10^-4 ... (0.5, 0.999)`
  - interpretation: `generator Adam lr=2e-4 betas=(0.5,0.999)`

- item_path: `attack.hyperparameters.lambda_div.cifar10`
  - pdf: `DFMS.pdf`
  - page: `15290`
  - ref: `Section 5 Ablation Study`
  - quote: `We set λdiv to 500 for CIFAR-10 and 100 for CIFAR-100`
  - interpretation: `lambda_div config values`

- item_path: `reported_results.tables[0].metrics.cifar10_resnet18_victim.dfms_hl_synthetic_acc`
  - pdf: `DFMS.pdf`
  - page: `15289`
  - ref: `Table 3`
  - quote: `Victim Accuracy∼93.7%, Victim Model: ResNet-18 ... DFMS-HL (Ours) ... 85.92`
  - interpretation: `paper target acc_gt=0.8592 for ResNet-18 victim row`

- item_path: `reported_results.tables[0].metrics.cifar10_resnet34_victim.dfms_sl_synthetic_acc`
  - pdf: `DFMS.pdf`
  - page: `15289`
  - ref: `Table 3`
  - quote: `DFMS-SL (Ours) ... 91.24`
  - interpretation: `soft-label variant reference point`
