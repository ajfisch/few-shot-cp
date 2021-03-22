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
    data = []
    for task in calibration:
        quantile = task["pred_quantile"]
        if task_type == "regression":
            # Query score is assumed to be |query_pred - query_target|.
            scores = task["query_scores"]
        else:
            # Query score is assumed to be score of correct class.
            # For now, we do "mondrian" conformal prediction (per class).
            scores = [[] for _ in range(len(quantile))]
            for s, t in zip(task["query_scores"], task["query_targets"]):
                scores[t].append(s[t])
            scores = np.array(scores)
        data.append((quantile, scores))

    if task_type == "regression":
        lambda_factor = conformalize_quantile(data, epsilon, delta)
    else:
        lambda_factor = []
        for i in range(len(data[0][0])):
            data_i = [(q[i], s[i]) for q, s in data]
            lambda_factor.append(conformalize_quantile(data_i, epsilon, delta))

    # Step 2: Compute intervals on test.
    conformal_pred = []
    if task_type == "regression":
        test_quantile = test["pred_quantile"] + lambda_factor
        for y in test["query_preds"]:
            conformal_pred.append((y - test_quantile, y + test_quantile))
    else:
        for label_scores in test["query_scores"]:
            kept_labels = []
            for i, v_y in enumerate(label_scores):
                test_quantile = test["pred_quantile"][i] + lambda_factor[i]
                if v_y <= test_quantile:
                    kept_labels.append(i)
            conformal_pred.append(kept_labels)

    return conformal_pred


def init(_tasks):
    global tasks
    tasks = _tasks


def evaluate_trial(task_idx, epsilon, delta, task_type, top_ks=[1,3,5]):
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
    if task_type == "regression":
        n_test = min(len(test["query_targets"]), len(test["query_encs"]))
    else:
        n_test = min(len(test["query_targets"]), len(test["exact_pred_scores"]))

    # Evaluate meta.
    meta_preds = compute_meta_predictions(
        calibration=calibration,
        test=test,
        epsilon=epsilon,
        delta=delta,
        task_type=task_type)
    meta_result = utils.evaluate(test["query_targets"], meta_preds, task_type, n_test)

    # Evaluate exact CP baseline.
    exact_preds = []
    topk_preds = {k: [] for k in top_ks}
    for i in range(n_test):
        if task_type == "classification":
            exact_pred = utils.exact_interval_classification(
                epsilon=epsilon,
                calibration=test["exact_non_comfs"][i],
                test=test["exact_pred_scores"][i])
            for k in top_ks:
                topk_preds[k].append(utils.topk_interval_classification(test["exact_pred_scores"][i], k=k))
        else:
            exact_pred = utils.exact_interval_regression(
                epsilon=epsilon,
                calibration=test["support_encs"],
                calibration_targets=test["support_targets"],
                test=test["query_encs"][i],
                l2_lambda=test["l2_lambda"])
        exact_preds.append(exact_pred)
    exact_result = utils.evaluate(test["query_targets"], exact_preds, task_type, n_test)
    topk_results = {}
    if task_type == "classification":
        for k in top_ks:
            topk_results[k] = utils.evaluate(test["query_targets"], topk_preds[k], task_type, n_test)

    return meta_result, exact_result, topk_results


def evaluate_trials(trials, tasks, epsilon, delta=0.9, task_type="classification", threads=1, top_ks=[1,3,5]):
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
    topk_results = {k: [] for k in top_ks}

    if task_type not in ["classification", "regression"]:
        raise ValueError("Unknown task type.")

    # Only use multiprocessing with threads > 1.
    if threads > 1:
        workers = multiprocessing.Pool(threads, initializer=init, initargs=(tasks,))
        map_fn = workers.imap_unordered
    else:
        init(tasks)
        map_fn = map

    # Setup trial evaluation function.
    worker_fn = functools.partial(
        evaluate_trial,
        epsilon=epsilon,
        delta=delta,
        task_type=task_type,
        top_ks=top_ks)

    # Map all results.
    with tqdm.tqdm(total=len(trials)) as pbar:
        for meta_result, exact_result, topk_result in map_fn(worker_fn, trials):
            meta_results.append(meta_result)
            exact_results.append(exact_result)
            if task_type == "classification":
                for k in top_ks:
                    topk_results[k].append(topk_result[k])
            pbar.update()

    # Marginal results.
    avg_meta_result = utils.compute_stats(meta_results)
    avg_exact_result = utils.compute_stats(exact_results)
    avg_topk_result = {k: utils.compute_stats(v) for k, v in topk_results.items()}

    aggr_results = dict(meta=avg_meta_result, exact=avg_exact_result)
    if task_type == "classification":
        for k in top_ks:
            aggr_results[f"top-{k}"] = avg_topk_result[k]

    return aggr_results
