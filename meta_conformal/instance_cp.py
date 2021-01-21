"""Compute experiment for instance-wise CP."""

import functools
import multiprocessing
import tqdm
import numpy as np
from scipy import stats
from meta_conformal import utils


def solve_correction(delta, ecdfs, bound="beta", steps=25):
    """Solve for correction giving (delta, epsilon)-validity."""

    def dkw(alpha, nobs, count):
        """Compute simultaneous Dvoretsky-Kiefer-Wolfowitz CDF bound."""
        if alpha == 0:
            return [0, 1]
        epsilon = np.sqrt(np.log(2 / alpha) / (2 * count))
        ci_lo = max(nobs / count - epsilon, 0)
        ci_hi = min(nobs / count + epsilon, 1)
        return ci_lo, ci_hi

    def clopper_pearson(alpha, nobs, count):
        """Compute pointwise Clopper-Pearson CDF bound."""
        if alpha == 0:
            return [0, 1]
        if nobs > 0:
            ci_lo = stats.beta.ppf(alpha / 2, nobs, count - nobs + 1)
        else:
            ci_lo = 0
        if nobs < count:
            ci_hi = stats.beta.ppf(1 - alpha / 2, nobs + 1, count - nobs)
        else:
            ci_hi = 1
        return ci_lo, ci_hi

    def correction(alpha):
        """Compute correction factor by inverting error bound."""
        if alpha == 1:
            return np.inf

        # Factor from joint prob. of n i.i.d. CDF errors in [a_i, b_i].
        p_bound = 1 - delta / np.power(1 - alpha, len(ecdfs))
        if p_bound <= 0:
            return np.inf

        # Factor from Hoeffding bound.
        bound_sum = 0
        for nobs, count in ecdfs:
            if bound == "dkw":
                a, b = dkw(alpha, nobs, count)
            elif bound == "beta":
                a, b = clopper_pearson(alpha, nobs, count)
            else:
                raise NotImplementedError("Unknown method.")
            bound_sum += (b - a) ** 2
        p_hoeffding = -bound_sum / (2 * len(ecdfs) ** 2)
        return np.sqrt(p_hoeffding * np.log(p_bound))

    # No need to compute.
    if delta == 0:
        return 0

    # Search for the best alpha in (0, max_alpha) using brute force.
    max_alpha = 1 - np.power(delta, 1 / len(ecdfs))
    best = np.inf
    for alpha in np.linspace(0, max_alpha, steps):
        t = correction(alpha)
        if t <= best:
            best = t

    return best


def conformalize_quantile(calibration_tasks, epsilon, delta=0):
    """Conformalize lambda for adding to quantile prediction."""

    def compute_ecdfs(lambda_factor):
        """Compute empirical CDF with lambda adjustment."""
        ecdfs = []
        for quantile, scores in calibration_tasks:
            nobs = (scores <= quantile + lambda_factor).sum()
            count = len(scores)
            ecdfs.append((nobs, count))
        return ecdfs

    def compute_bound(lambda_factor):
        """Compute full bound."""
        ecdfs = compute_ecdfs(lambda_factor)
        n_tasks = len(calibration_tasks) + 1
        bound = sum([nobs / count for nobs, count in ecdfs]) / n_tasks
        bound -= solve_correction(delta, ecdfs)
        return bound

    # Compute with no correction.
    lambda_factor = 0
    bound = compute_bound(lambda_factor)
    if bound == 1 - epsilon:
        return lambda_factor

    # Get range of lambda that change empirical CDFs.
    valid_lambdas = [np.inf]
    for quantile, scores in calibration_tasks:
        rank = (scores <= quantile).sum()
        if bound < 1 - epsilon:
            # Add positive lambdas.
            for i in range(rank, len(scores)):
                valid_lambdas.append(scores[i] - scores[rank - 1])
        else:
            # Add negative lambdas.
            for i in range(rank - 1, -1, -1):
                valid_lambdas.append(scores[rank - 1] - scores[i])
    valid_lambdas = list(set(valid_lambdas))

    # Binary search lambdas to find inf.
    valid_lambdas = sorted(valid_lambdas)
    best_lambda = np.inf
    left = 0
    right = len(valid_lambdas) - 1
    while left <= right:
        # Compute empirical CDFs (with lambda adjustment).
        mid = (left + right) // 2
        lambda_factor = valid_lambdas[mid]

        # Compute full bound (with prob. delta).
        bound = compute_bound(lambda_factor)

        # Save if best valid so far.
        if bound >= 1 - epsilon and lambda_factor < best_lambda:
            best_lambda = lambda_factor

        # Continue search.
        if bound == 1 - epsilon:
            break
        elif bound < 1 - epsilon:
            left = mid + 1
        else:
            right = mid - 1

    return best_lambda


def compute_meta_predictions(calibration, test, epsilon, delta=0, task_type="classification"):
    """Evaluate a meta conformal trial over task instances.

    Args:
        calibration: Calibration tasks (1 ... N).
        test: Test task (N + 1).
        epsilon: epsilon-validity.
        delta: (delta, epsilon)-validity.
        task_type: Either "classification" or "regression".

    Returns:
        result: average efficiency and accuracy for task N + 1.
    """
    # Step 1: conformalize the quantile prediction on calibration tasks.
    calibration = [(task["pred_quantile"], task["query_scores"]) for task in calibration]
    lambda_factor = conformalize_quantile(calibration, epsilon, delta)
    test_quantile = test["pred_quantile"] + lambda_factor

    # Step 2: Compute intervals on test.
    conformal_pred = []
    if task_type == "regression":
        for y in test["query_preds"]:
            conformal_pred.append((y - test_quantile, y + test_quantile))
    elif task_type == "classification":
        for label_scores in test["query_scores"]:
            kept_labels = []
            for i, v_y in enumerate(label_scores):
                if v_y <= test_quantile:
                    kept_labels.append(i)
            conformal_pred.append(kept_labels)
    else:
        raise ValueError("Unknown type %s" % task_type)

    return conformal_pred


def init(_tasks):
    global tasks
    tasks = _tasks


def evaluate_trial(task_idx, epsilon, delta, task_type):
    """Worker function."""
    global tasks

    # Get data.
    task_samples = []
    for task_i, example_j in task_idx:
        task_samples.append(tasks[task_i][example_j])

    # Get calibration/test task split.
    calibration, test = task_samples[:-1], task_samples[-1]

    # We may have only computed metrics on a subset of full calibration data...
    # take only up to the first n_test.
    n_test = min(len(test["query_targets"]), len(test["exact_intervals"]))

    # Evaluate uncorrected meta.
    meta_preds = compute_meta_predictions(
        calibration=calibration,
        test=test,
        epsilon=epsilon,
        delta=delta,
        task_type=task_type)
    meta_result = utils.evaluate(test["query_targets"], meta_preds, task_type, n_test)

    # Evaluate exact CP baseline.
    exact_preds = test["exact_intervals"]
    exact_result = utils.evaluate(test["query_targets"], exact_preds, task_type, n_test)

    return meta_result, exact_result


def evaluate_trials(trials, tasks, epsilon, delta=0.9, task_type="classification", threads=1):
    """Evaluate conformal prediction over collection of N + 1 tasks.

    Args:
        trials: List[(task index, k-shot index)].
        tasks: List of tasks with multiple k-shot instances.
        epsilon: epsilon-validity.
        delta: (delta, epsilon)-validity.
        task_type: Either "classification" or "regression".
    """
    meta_results = []
    exact_results = []

    # Only use multiprocessing with threads > 1.
    if threads > 0:
        workers = multiprocessing.Pool(threads, initializer=init, initargs=(tasks,))
        map_fn = workers.imap_unordered
    else:
        map_fn = map

    # Setup trial evaluation function.
    worker_fn = functools.partial(
        evaluate_trial,
        epsilon=epsilon,
        delta=delta,
        task_type=task_type)

    # Map all results.
    with tqdm.tqdm(total=len(trials)) as pbar:
        for meta_result, exact_result in map_fn(worker_fn, trials):
            meta_results.append(meta_result)
            exact_results.append(exact_result)
            pbar.update()

    # Marginal results.
    avg_meta_result = utils.compute_stats(meta_results)
    avg_exact_result = utils.compute_stats(exact_results)

    return dict(meta=avg_meta_result, exact=avg_exact_result)
