"""Conformal prediction utilities."""

import collections
import numpy as np
import intervals

Result = collections.namedtuple("Result", ["eff", "acc"])


def compute_stats(results):
    """Compute efficiency and accuracy statistics."""

    def _stats(_results):
        mean = np.mean(_results)
        std = np.std(_results)
        p_84 = np.percentile(_results, 84)
        p_16 = np.percentile(_results, 16)
        return (mean, std, p_84, p_16)

    eff_stats = _stats([r.eff for r in results])
    acc_stats = _stats([r.acc for r in results])

    return Result(eff_stats, acc_stats)


def evaluate(y_trues, y_preds, task_type="classification"):
    """Evaluate sequence of predictions and confidence intervals."""
    evals = []
    sizes = []
    for y_true, y_pred in zip(y_trues, y_preds):
        if task_type == "regression":
            y_pred = intervals.closed(*y_pred)
            size = y_pred.upper - y_pred.lower
        elif task_type == "classification":
            size = len(y_pred)
        else:
            raise ValueError("Unknown type %s" % task_type)
        is_correct = y_true in y_pred
        evals.append(is_correct)
        sizes.append(size)

    return Result(np.mean(sizes), np.mean(evals))
