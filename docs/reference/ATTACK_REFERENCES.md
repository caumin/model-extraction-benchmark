# Attack References and Provenance

This document records, for each implemented attack, (1) the proposing paper and
(2) whether the mebench implementation was ported from an official open-source
repository.

Legend:
- `Paper-only`: implemented from paper/specs without direct code port from an official repo clone.
- `Official repo port`: implementation adapted from an official/open-source repository.
- `Benchmark native`: benchmark baseline implementation (no proposing paper or official repo port required).

## Reference Table

| Attack | Proposing paper | Provenance | Official repository | Official source entrypoint (GitHub) | mebench implementation |
|---|---|---|---|---|---|
| RandomBaseline | - | Benchmark native | - | - | `mebench/attackers/random_baseline.py` |
| ActiveThief | https://ojs.aaai.org/index.php/AAAI/article/view/5432 | Official repo port | https://github.com/iisc-seal/activethief | https://github.com/iisc-seal/activethief/blob/master/generic_program.py | `mebench/attackers/activethief.py` |
| BlackboxDissector | https://arxiv.org/abs/2105.00623 | Official repo port | https://github.com/yxwang-10/Blackbox-Dissector | https://github.com/yxwang-10/Blackbox-Dissector/blob/main/attack.py | `mebench/attackers/blackbox_dissector.py` |
| CloudLeak | https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf | Official repo port | https://github.com/yunyuntsai/DNN-Model-Stealing | https://github.com/yunyuntsai/DNN-Model-Stealing/blob/main/optimize.py | `mebench/attackers/cloudleak.py` |
| InverseNet | https://www.ijcai.org/proceedings/2021/336 | Paper-only | - | - | `mebench/attackers/inversenet.py` |
| SwiftThief | https://www.ijcai.org/proceedings/2024/47 | Official repo port | https://github.com/ku-air/SwiftThief | https://github.com/ku-air/SwiftThief/blob/main/swiftthief.py | `mebench/attackers/swiftthief.py` |
| DFME | https://arxiv.org/abs/2011.14779 | Official repo port | https://github.com/cake-lab/datafree-model-extraction | https://github.com/cake-lab/datafree-model-extraction/blob/main/dfme/train.py | `mebench/attackers/dfme.py` |
| DFMS / DFMSHL | https://arxiv.org/abs/2204.11022 | Official repo port | https://github.com/val-iisc/Hard-Label-Model-Stealing | https://github.com/val-iisc/Hard-Label-Model-Stealing/blob/main/code/train_student/train_student.py | `mebench/attackers/dfms.py` |
| DisGUIDE | https://ojs.aaai.org/index.php/AAAI/article/view/26150 | Official repo port | https://github.com/lin-tan/disguide | https://github.com/lin-tan/disguide/blob/main/disguide/train.py | `mebench/attackers/disguide.py` |
| MAZE | https://arxiv.org/abs/2005.03161 | Official repo port | https://github.com/sanjaykariyappa/MAZE | https://github.com/sanjaykariyappa/MAZE/blob/master/src/attacks/maze.py | `mebench/attackers/maze.py` |
| ESAttack | https://arxiv.org/abs/2009.09560 | Paper-only | - | - | `mebench/attackers/es_attack.py` |
| GAME | https://link.springer.com/chapter/10.1007/978-3-031-17140-6_28 | Official repo port | https://github.com/xythink/game-attack | https://github.com/xythink/game-attack/blob/main/attack.py | `mebench/attackers/game.py` |
| KnockoffNets | https://arxiv.org/abs/1812.02766 | Official repo port | https://github.com/tribhuvanesh/knockoffnets | https://github.com/tribhuvanesh/knockoffnets/blob/master/knockoff/adversary/transfer.py | `mebench/attackers/knockoff_nets.py` |
| CopycatCNN | https://arxiv.org/abs/1806.05476 | Paper-only | https://github.com/jeiks/Stealing_DL_Models | https://github.com/jeiks/Stealing_DL_Models/blob/master/Copycat_CNN/Scripts/label_dataset.py | `mebench/attackers/copycatcnn.py` |
| BlackboxRipper | https://arxiv.org/abs/2010.11158 | Paper-only | https://github.com/antoniobarbalau/black-box-ripper | https://github.com/antoniobarbalau/black-box-ripper/blob/master/base_experiment.py | `mebench/attackers/blackbox_ripper.py` |
| DualStudents (DS) | https://arxiv.org/abs/2309.10058 | Official repo port | https://github.com/James-Beetham/dual_students | https://github.com/James-Beetham/dual_students/blob/main/dual_students/train.py | `mebench/attackers/ds.py` |
| MARICH | https://arxiv.org/abs/2302.08466 | Official repo port | https://github.com/Debabrota-Basu/marich | https://github.com/Debabrota-Basu/marich/tree/main/lr_cnn_res_marich | `mebench/attackers/marich.py` |

## Notes

- Source-of-truth for internal mapping status is maintained in:
  - `mebench/attackers/ATTACK_PARITY_REPORT.md`
  - `mebench/attackers/PORTING_PLAN.md`
  - `mebench/attackers/HYPERPARAM_POLICY.md`
- Matrix-generation policy (`generate_configs.py`) for SET-B currently uses two paths:
  - **Reference-aligned path** (optimizer + lr-per-sample + batch=512) for attacks with sufficiently comparable paper/official settings in the image-classification + resnet18 setup.
  - **Heuristic-default path** for attacks where source settings are missing, inconsistent, or not directly comparable for this exact setup.
- Current SET-B reference-aligned attacks: `dfme`, `disguide`, `ds`, `maze`, `knockoff_nets`, `blackbox_dissector`, `blackbox_ripper`, `swiftthief`.
- Current SET-B heuristic-default attacks (subject to future refinement): `activethief`, `marich`, `cloudleak`, `inversenet`, `random`, `dfms`.
- "Official repository" can be listed as a provenance reference even when `Provenance` is `Paper-only`; direct code-port status is determined by the `Provenance` column.
- Public GitHub links are provided so provenance remains verifiable even when local clone directories are excluded from release.
- MARICH currently keeps its original staged paper protocol; full harmonization with the benchmark's unified pool-based protocol is a planned follow-up.
