"""Run the conformal prediction experiments."""

import argparse
import collections
import functools
import json
import multiprocessing
import numpy as np
import os
import tqdm
import intervals as I

Result = collections.namedtuple("Result", ["eff", "acc"])


def compute_stats(results):
    def _stats(_results):
        mean = np.mean(_results)
        p_84 = np.percentile(_results, 84)
        p_16 = np.percentile(_results, 16)
        return (mean, p_84, p_16)
    eff_stats = _stats([r.eff for r in results])
    acc_stats = _stats([r.acc for r in results])

    return dict(eff=eff_stats, acc=acc_stats)


def load_trials(dataset_file, num_trials):
    """Load data and sample into trials."""
    task2idx = {}
    idx2examples = collections.defaultdict(list)

    # Read examples and partition to tasks.
    with open(dataset_file, "r") as f:
        for line in f:
            example = json.loads(line)
            if example["task"] not in task2idx:
                task2idx[example["task"]] = len(task2idx)
            idx = task2idx[example["task"]]
            idx2examples[idx].append(example)
    idx2examples = [idx2examples[i] for i in range(len(idx2examples))]

    num_tasks = len(task2idx)
    print("Number of tasks: %d" % num_tasks)

    episodes_per_task = np.mean([len(v) for v in idx2examples])
    print("Avg number of episodes per task: %2.2f" % episodes_per_task)

    # For every trial, we sample a random order of tasks (i.e. calibration vs test).
    trials = []
    for _ in range(num_trials):
        order = np.random.permutation(num_tasks)
        trials.append(order)

    return trials, idx2examples


def conformalize_quantile(predictions, calibration, alpha, beta):
    """
    predictions: [num_tasks]
    calibration: [num_tasks, num_calibration]
    alpha: scalar
    beta: scalar
    """
    errors = []
    for t, task_predictions in enumerate(predictions):
        sample_errors = []
        for s, prediction in enumerate(task_predictions):
            true_quantile = np.quantile(calibration[t][s] + [np.inf], alpha, interpolation="higher")
            sample_errors.append(true_quantile - prediction)
        errors.append(np.mean(sample_errors))
    delta = np.quantile(errors + [np.inf], beta, interpolation="higher")
    return delta


def optimize_conformal_quantile(predictions, calibration, epsilon):
    """Pick the conformalizing delta for the quantile predictor."""
    t = len(predictions)
    i = 0
    min_delta = np.inf
    while i / (t + 1) <= epsilon:
        beta = 1 - i / (t + 1)
        alpha = (1 - epsilon) / beta
        delta = conformalize_quantile(predictions, calibration, alpha, beta)
        min_delta = min(min_delta, delta)
        i += 1
    return min_delta


def evaluate_trial(tasks, num_samples, epsilon):
    """Evaluate a conformal trial."""
    global idx2examples

    task_examples = []
    for i in tasks:
        examples = idx2examples[i]
        samples = np.random.permutation(len(examples))[:100]
        task_examples.append([examples[j] for j in samples])
    tasks = task_examples

    assert(len(tasks[0][0]["exact_intervals"]) >= num_samples)
    assert(len(tasks[0][0]["jk_intervals"]) >= num_samples)

    # Step 1: Conformalize the quantile on calibration tasks.
    predictions = [[ex["pred_quantile"] for ex in t] for t in tasks[:-1]]
    calibration = [[[abs(x - y) for x, y in zip(ex["query_preds"], ex["query_targets"])] for ex in t]
                   for t in tasks[:-1]]
    delta = optimize_conformal_quantile(predictions, calibration, epsilon)

    # Step 2: Compute intervals on the test task.
    eff = {"meta": [], "exact": [], "jackknife": []}
    acc = {"meta": [], "exact": [], "jackknife": []}

    for example in tasks[-1]:
        quantile = example["pred_quantile"] + delta
        test_preds = example["query_preds"][:num_samples]
        test_targets = example["query_targets"][:num_samples]
        meta_intervals = [I.closed(p - quantile, p + quantile) for p in test_preds]

        exact_intervals = [I.closed(*interval) for interval in example["exact_intervals"][:num_samples]]
        jk_intervals = [I.closed(*interval) for interval in example["jk_intervals"][:num_samples]]

        eff["meta"].append(2 * quantile)
        acc["meta"].append(np.mean([t in interval for interval, t in zip(meta_intervals, test_targets)]))

        eff["exact"].append(np.mean([interval.upper - interval.lower for interval in exact_intervals]))
        acc["exact"].append(np.mean([t in interval for interval, t in zip(exact_intervals, test_targets)]))

        eff["jackknife"].append(np.mean([interval.upper - interval.lower for interval in jk_intervals]))
        acc["jackknife"].append(np.mean([t in interval for interval, t in zip(jk_intervals, test_targets)]))

    meta_result = Result(np.mean(eff["meta"]), np.mean(acc["meta"]))
    exact_result = Result(np.mean(eff["exact"]), np.mean(acc["exact"]))
    jackknife_result = Result(np.mean(eff["jackknife"]), np.mean(acc["jackknife"]))

    return meta_result, exact_result, jackknife_result


def worker_init(_idx2examples):
    global idx2examples
    idx2examples = _idx2examples


def main(args):
    np.random.seed(42)

    # Load data.
    print("Loading data...")
    trials, idx2examples = load_trials(args.dataset_file, args.num_trials)

    # Initialize workers.
    worker_fn = functools.partial(
        evaluate_trial,
        num_samples=args.num_trial_samples,
        epsilon=args.epsilon)
    if args.threads > 0:
        workers = multiprocessing.Pool(args.threads, initializer=worker_init, initargs=(idx2examples,))
        map_fn = workers.imap_unordered
    else:
        map_fn = map
        worker_init(idx2examples)

    # --------------------------------------------------------------------------
    # Step 1.
    #
    # Conformalize the quantile predictor. This is marginal over task examples.
    # --------------------------------------------------------------------------

    # Evaluate!
    meta_results = []
    exact_results = []
    jk_results = []
    with tqdm.tqdm(total=len(trials), desc="evaluating") as pbar:
        for cp, exact, jk in map_fn(worker_fn, trials):
            meta_results.append(cp)
            exact_results.append(exact)
            jk_results.append(jk)
            pbar.update()

    # Report statistics.
    meta_stats = compute_stats(meta_results)
    exact_stats = compute_stats(exact_results)
    jk_stats = compute_stats(jk_results)
    results = {"meta": meta_stats, "exact": exact_stats, "jackknife": jk_stats}
    for k, v in results.items():
        print("=" * 50)
        print("Method: %s" % k)
        for kk, vv in v.items():
            print("%s\t%s" % (kk, vv))
        print("")

    if args.results_file is None:
        basename = os.path.splitext(args.dataset_file)[0]
        args.results_file = basename + ".trials=%d.results.json" % args.num_trials

    print("Writing to %s" % args.results_file)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="../ckpts/chembl/k=10/conformal/q=0.73/val.jsonl")
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--num_trial_samples", type=int, default=4)
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.27)
    parser.add_argument("--threads", type=int, default=40)
    args = parser.parse_args()
    main(args)
