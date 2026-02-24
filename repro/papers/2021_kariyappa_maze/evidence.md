# Evidence Log

- item_path: `paper.title`
  - pdf: `MAZE.pdf`
  - page: `13814`
  - ref: `Title`
  - quote: `MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation`
  - interpretation: `paper_meta/extracted_spec title`

- item_path: `attack.oracle_access.type`
  - pdf: `MAZE.pdf`
  - page: `13816`
  - ref: `Section 3 Preliminaries`
  - quote: `soft-label setting where the adversary can query ... and observe its output probabilities`
  - interpretation: `oracle output type = soft probabilities`

- item_path: `attack.hyperparameters.grad_approx_epsilon`
  - pdf: `MAZE.pdf`
  - page: `13818`
  - ref: `Section 4.4`
  - quote: `We set the value of ε to 0.001 in our experiments`
  - interpretation: `grad_approx_epsilon=0.001`

- item_path: `attack.hyperparameters.batch_size`
  - pdf: `MAZE.pdf`
  - page: `13819`
  - ref: `Section 4.5`
  - quote: `We use B = 128, NG = 1, NC = 5, NR = 10 and m = 10`
  - interpretation: `batch_size=128, n_g=1, n_c=5, n_r=10, m=10`

- item_path: `victim.architecture.name`
  - pdf: `MAZE.pdf`
  - page: `13819`
  - ref: `Section 5.1`
  - quote: `We use a LeNet for the FashionMNIST and ResNet-20 for the other datasets`
  - interpretation: `victim architecture mapping`

- item_path: `attack.substitute.architecture`
  - pdf: `MAZE.pdf`
  - page: `13819`
  - ref: `Section 5.1`
  - quote: `uses a randomly initialized 22-layer WideResNet as the clone model`
  - interpretation: `substitute architecture = wideresnet22`

- item_path: `attack.query_budget.cifar10`
  - pdf: `MAZE.pdf`
  - page: `13819`
  - ref: `Section 4.5`
  - quote: `query budget of 30M for GTSRB and CIFAR-10 datasets`
  - interpretation: `query_budget.cifar10=30000000`

- item_path: `reported_results.tables[0].metrics.cifar10.maze_acc`
  - pdf: `MAZE.pdf`
  - page: `13820`
  - ref: `Table 1`
  - quote: `CIFAR-10 ... 89.85 (0.97×)`
  - interpretation: `paper target acc_gt=0.8985`
