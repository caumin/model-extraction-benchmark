# Evaluation Metrics

All metrics are computed on the task's public test set with the victim's
official evaluation normalization (see `mebench/data/preprocessing.py`).
Implementations are in `mebench/eval/metrics.py`.

## Accuracy

Top-1 accuracy of the substitute on the public test set:

```
acc = mean( argmax(student(x)) == y )
```

For single-logit binary classifiers, predictions are taken via
`binary_hard_labels_from_logits`. See `compute_accuracy` at
`mebench/eval/metrics.py:18`.

## Agreement

Probability that the substitute and victim predict the same top-1 label on
the public test set:

```
agreement = mean( argmax(student(x)) == argmax(victim(x)) )
```

Reported per-checkpoint to track functional fidelity. See
`compute_agreement` at `mebench/eval/metrics.py:46`.

## KL divergence

Per-sample mean of `KL(softmax(student(x)) || softmax(victim(x)))` at
temperature `T = 1.0` by default:

```
kl = mean_x  sum_c  P_v(c|x) * ( log P_v(c|x) - log P_s(c|x) )
```

where `P_s = softmax(student(x) / T)` and `P_v = softmax(victim(x) / T)`. We
use `F.kl_div(log(P_s + 1e-10), P_v, reduction='sum')` per batch and divide
by sample count. See `compute_kl_divergence` at `mebench/eval/metrics.py:77`.

Note: this computes `KL(student || victim)` in F.kl_div's argument order
convention, which matches the standard student-distillation formulation.

## L1 distance

Per-class average of `|P_s - P_v|`:

```
l1 = mean_x  ( sum_c |P_v(c|x) - P_s(c|x)| ) / num_classes
```

See `compute_l1_distance` at `mebench/eval/metrics.py:116`.

## Reporting convention

- Accuracy and agreement are the headline numbers for both pool-based and
  data-free attacks.
- KL / L1 are reported for soft-prob attacks only and are informative for
  distillation quality.
- All numbers are reported per checkpoint at the budget thresholds declared
  in `budget.checkpoints`.
- Within-set comparisons are the primary fairness target; cross-set numbers
  are contextual (see `THREAT_MODEL.md`).
