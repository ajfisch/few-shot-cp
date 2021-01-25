"""Conformal prediction utilities."""

import collections
import json
import numpy as np
import intervals
from meta_conformal import rrcm

Result = collections.namedtuple("Result", ["eff", "acc"])


def compute_stats(results):
    """Compute efficiency and accuracy statistics."""

    def _stats(_results):
        if any(np.isinf(r) for r in _results):
            return (np.inf, np.inf, np.inf, np.inf)
        mean = np.mean(_results)
        std = np.std(_results)
        p_84 = np.percentile(_results, 84)
        p_16 = np.percentile(_results, 16)
        return (mean, std, p_84, p_16)

    eff_stats = _stats([r.eff for r in results])
    acc_stats = _stats([r.acc for r in results])

    return Result(eff_stats, acc_stats)


def evaluate(y_trues, y_preds, task_type="classification", n_test=None):
    """Evaluate sequence of predictions and confidence intervals."""
    evals = []
    sizes = []
    for i, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        if n_test and i >= n_test:
            break
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


def write_result(epsilon, delta, result, filename, overwrite=False):
    """Write result to file."""
    if overwrite:
        mode = "w"
    else:
        mode = "a"
    with open(filename, mode) as f:
        result = {k: dict(v._asdict()) for k, v in result.items()}
        result = {"epsilon": epsilon, "delta": delta, "result": result}
        f.write(json.dumps(result) + "\n")


def exact_interval_regression(epsilon, calibration, calibration_targets, test, l2_lambda):
    """Compute RRCM exact conformal interval.

    Args:
        epsilon: Tolerance.
        calibration: [n_support, n_dim]
        calibration_targets: [n_support]
        test: [n_dim]
        l2_lambda: Regularization parameter.

    Returns:
       interval: [lower, higher]
    """
    X = np.concatenate([calibration, np.expand_dims(test, 0)], axis=0)
    Y_ = calibration_targets
    return rrcm.conf_pred(X, Y_, l2_lambda, epsilon)


def exact_interval_classification(epsilon, calibration, test):
    """Compute full conformal prediction interval.

    Args:
        epsilon: Tolerance.
        calibration: [n_label, n_support]
        test: [n_label]

    Returns:
       interval: {y : y <= quantile}
    """
    pred_set = []
    for y, y_score in enumerate(test):
        scores = np.concatenate([calibration[y], [np.inf]])
        quantile = np.quantile(scores, 1 - epsilon, interpolation="higher")
        if y_score <= quantile:
            pred_set.append(y)
    return pred_set
