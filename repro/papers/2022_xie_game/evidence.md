# Evidence Log

- item_path: `datasets(pair-1)`
  - pdf: `GAME.pdf`
  - page: `11`
  - ref: `Section 4.1 Datasets and victim model`
  - quote: `MNIST ... original dataset and Fashion-MNIST ... proxy dataset`
  - interpretation: `primary repro pair set to MNIST/FashionMNIST`

- item_path: `victim.training(pair-1)`
  - pdf: `GAME.pdf`
  - page: `11`
  - ref: `Section 4.1 Datasets and victim model`
  - quote: `victim models were trained for 15 epochs ... with ADAM at an initial learning rate of 0.001`
  - interpretation: `victim_train config uses epochs=15, optimizer=adam, lr=0.001`

- item_path: `attack.query_budget`
  - pdf: `GAME.pdf`
  - page: `11`
  - ref: `Section 4.1 Attacker model`
  - quote: `The query budget is 8k for Fashion-MNIST and 6k for GTSRB`
  - interpretation: `full repro config uses 8k pair-1 budget`

- item_path: `attack training defaults`
  - pdf: `GAME.pdf`
  - page: `11`
  - ref: `Section 4.1 Attacker model`
  - quote: `These models were trained for 40 epochs with ADAM. The initial learning rate is 0.1 for half-LeNet and 0.01 for VGG-16.`
  - interpretation: `attack_train_epoch=40 and optimizer=adam in GAME attack config`

- item_path: `reported_results.table1.game_half_lenet`
  - pdf: `GAME.pdf`
  - page: `12`
  - ref: `Table 1`
  - quote: `GAME (Ours) ... Fidelity 90.93 ... Accuracy 90.36 ... Relative 0.92`
  - interpretation: `pair-1 primary target values captured in extracted_spec`

- item_path: `reported_results.table2.game_resnet18`
  - pdf: `GAME.pdf`
  - page: `13`
  - ref: `Table 2`
  - quote: `GAME (Ours) ... Fidelity 74.52 ... Accuracy 73.77 ... Relative 0.75`
  - interpretation: `pair-2 reference values captured in extracted_spec`
