# Evidence Log

- item_path: `paper.title`
  - pdf: `DFME.pdf`
  - page: `4771`
  - ref: `Title`
  - quote: `Data-Free Model Extraction`
  - interpretation: `paper_meta/extracted_spec title`

- item_path: `attack.query_budget.svhn`
  - pdf: `DFME.pdf`
  - page: `4776`
  - ref: `Section 5.1 Datasets and Architectures`
  - quote: `The default query budget Q is 2M for SVHN`
  - interpretation: `query_budget.svhn=2000000`

- item_path: `attack.query_budget.cifar10`
  - pdf: `DFME.pdf`
  - page: `4776`
  - ref: `Section 5.1 Datasets and Architectures`
  - quote: `and 20M for CIFAR-10 in our experiments`
  - interpretation: `query_budget.cifar10=20000000`

- item_path: `victim.architecture.name`
  - pdf: `DFME.pdf`
  - page: `4776`
  - ref: `Section 5.1 Datasets and Architectures`
  - quote: `the victim model architecture is a ResNet-34-8x`
  - interpretation: `victim.architecture.name=ResNet-34-8x`

- item_path: `attack.substitute.architecture`
  - pdf: `DFME.pdf`
  - page: `4776`
  - ref: `Section 5.1 Datasets and Architectures`
  - quote: `We use ResNet-18-8x as the architecture for our student model`
  - interpretation: `substitute architecture ResNet-18-8x`

- item_path: `attack.hyperparameters.n_g_steps`
  - pdf: `DFME.pdf`
  - page: `4774`
  - ref: `Algorithm 1`
  - quote: `Generator iters n_G, student iters n_S`
  - interpretation: `n_g_steps=1, n_s_steps=5 (paper defaults)`

- item_path: `attack.hyperparameters.grad_approx_m`
  - pdf: `DFME.pdf`
  - page: `4776`
  - ref: `Section 5.1 Datasets and Architectures`
  - quote: `sample m = 1 random directions and a step size epsilon = 10^-3`
  - interpretation: `grad_approx_m=1, grad_approx_epsilon=0.001`

- item_path: `reported_results.tables[0].metrics.cifar10.dfme_acc`
  - pdf: `DFME.pdf`
  - page: `4777`
  - ref: `Table 1`
  - quote: `CIFAR10 (20M) ... DFME 88.1% (0.92x)`
  - interpretation: `paper target acc_gt=0.881 at 20M`

- item_path: `reported_results.tables[0].metrics.svhn.dfme_acc`
  - pdf: `DFME.pdf`
  - page: `4777`
  - ref: `Table 1`
  - quote: `SVHN (2M) ... DFME 95.2% (0.99x)`
  - interpretation: `paper target acc_gt=0.952 at 2M`
