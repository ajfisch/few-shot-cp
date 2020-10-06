"""Experiment for CP results."""

import argparse
import collections
import json
import math
import numpy as np
import subprocess
import tqdm


def stats(metrics):
    return {
        "avg": np.mean(metrics),
        "std": np.std(metrics),
    }


def load(filename, quantile):
    # task --> few shot examples.:
    data = collections.defaultdict(list)
    num_lines = int(subprocess.check_output(["wc", "-l", filename], encoding="utf8").split()[0])
    with open(filename, "r") as f:
        i = 0
        for line in tqdm.tqdm(f, total=num_lines, desc="reading lines"):
            i += 1
            entry = json.loads(line)
            if i > 1000:
                break

            # Add true quantile before modifying output scores.
            entry["true_quantile"] = np.quantile(entry["output_scores"] + [float("inf")], quantile)
            entry["cdf_scores"] = np.array(sorted(entry["output_scores"] + [float("inf")]))

            # Add complement to output_scores.
            output_scores = []
            for score, label in zip(entry["output_scores"], entry["output_labels"]):
                if label == 0:
                    negative = np.exp(-score)
                    positive = 1 - negative
                else:
                    positive = np.exp(-score)
                    negative = 1 - positive
                output_scores.append([-np.log(negative), -np.log(positive)])
            entry["output_scores"] = output_scores

            # Add complement to cv_scores.
            cv_scores = []
            for scores, labels in zip(entry["cv_scores"], entry["cv_labels"]):
                cv_fold = []
                for score, label in zip(scores, labels):
                    if label == 0:
                        negative = np.exp(-score)
                        positive = 1 - negative
                    else:
                        positive = np.exp(-score)
                        negative = 1 - positive
                    cv_fold.append([-np.log(negative), -np.log(positive)])
                cv_scores.append(cv_fold)
            entry["cv_scores"] = cv_scores

            data[entry["task"]].append(entry)
    return data


def compute_baseline(data, epsilon, use_correction=False):
    example = next(iter(data.values()))[0]
    K = len(example["input_scores"])
    n = len([x for y in example["input_scores"] for x in y])
    if use_correction:
        factor_1 = (1 - 1 / K) / (n / K + 1)
        factor_2 = 0.5 * (1 - K / n) / (K + 1)
        epsilon = epsilon / 2 - min(factor_1, factor_2)

    efficiencies = []
    accuracies = []

    # Run through each of the meta-tasks.
    for task, few_shot_trials in data.items():

        # Store task accuracies and efficies.
        task_accuracy = []
        task_efficiency = []

        # For each meta-task, evaluate each of the iterators of few-shot prediction.
        for trial in few_shot_trials:

            # Store trial accuracies and efficiencies.
            accuracy = []
            efficiency = []

            # Each few-shot iteration is evaluated over multiple query samples.
            for i in range(len(trial["output_scores"])):
                label = trial["output_labels"][i]
                prediction = []

                # Check each possible label.
                for y in [0, 1]:
                    # Aggregated over folds, count the number of calibration points that are
                    # more conforming than the query point. If this is less than 1 - epsilon,
                    # then the point does *not* fall in the epsilon nonconformal quantile,
                    # and we keep it.
                    num_more_conformal = 0
                    for input_fold, cv_fold in zip(trial["input_scores"], trial["cv_scores"]):
                        for score in input_fold:
                            if score < cv_fold[i][y]:
                                num_more_conformal += 1
                    if num_more_conformal < (1 - epsilon) * (n + 1):
                        prediction.append(y)

                # Compute accuracy and efficiency.
                accuracy.append(float(label in prediction))
                efficiency.append(len(prediction))

            # Update task metrics.
            task_accuracy.append(np.mean(accuracy))
            task_efficiency.append(np.mean(efficiency))

        # Update meta metrics.
        accuracies.append(np.mean(task_accuracy))
        efficiencies.append(np.mean(task_efficiency))

    accuracy_stats = stats(accuracies)
    efficiency_stats = stats(efficiencies)

    return dict(accuracy=accuracy_stats, efficiency=efficiency_stats)


def compute_meta_conformal_conservative(calibration, test, epsilon, bootstrap=False, use_correction=False):
    # Decide appropriate calibration values based on bootstrap or not.
    alpha = np.sqrt(1 - epsilon)
    if bootstrap:
        if use_correction:
            beta = (1 + alpha) / 2
        else:
            beta = alpha
        n = max([len(trials) for trials in calibration.values()])
    else:
        beta = alpha
        n = 1

    # First, find the conformal adjustment over the calibration data for quantile error.
    # The error is positive only if we *underestimate* the quantile.
    bootstrap_errors = []
    for _ in range(n):
        errors = []
        mse = []
        for task, few_shot_trials in calibration.items():
            trial_idx = np.random.choice(len(few_shot_trials))
            trial = few_shot_trials[trial_idx]
            pred_quantile = trial["quantile"]
            true_quantile = trial["true_quantile"]
            errors.append(max(0, true_quantile - pred_quantile))
            mse.append((true_quantile - pred_quantile) ** 2)
        bootstrap_errors.append(np.quantile(errors + [float("inf")], beta))
    error_quantile = np.mean(bootstrap_errors)

    # Next, iterate through the test data, using the inflated quantile.
    efficiencies = []
    accuracies = []

    # Run through each of the meta-tasks.
    for task, few_shot_trials in test.items():

        # Store task accuracies and efficienciess.
        task_accuracy = []
        task_efficiency = []

        # For each meta-task, evaluate each of the iterators of few-shot prediction.
        for trial in few_shot_trials:

            # Store trial accuracies and efficiencies.
            accuracy = []
            efficiency = []

            # Each few-shot iteration is evaluated over multiple query samples.
            for i in range(len(trial["output_scores"])):
                label = trial["output_labels"][i]
                quantile = trial["quantile"]
                prediction = []

                # Check each possible label.
                for y in [0, 1]:
                    score = trial["output_scores"][i][y]
                    if score <= quantile + error_quantile:
                        prediction.append(y)

                # Compute accuracy and efficiency.
                accuracy.append(float(label in prediction))
                efficiency.append(len(prediction))

            # Update task metrics.
            task_accuracy.append(np.mean(accuracy))
            task_efficiency.append(np.mean(efficiency))

        # Update meta metrics.
        accuracies.append(np.mean(task_accuracy))
        efficiencies.append(np.mean(task_efficiency))

    accuracy_stats = stats(accuracies)
    efficiency_stats = stats(efficiencies)

    return dict(accuracy=accuracy_stats, efficiency=efficiency_stats)


def compute_meta_conformal(calibration, test, epsilon):
    # Target coverage level.
    target = 1 - epsilon

    # Number of calibration tasks.
    n = len(calibration)
    m = len(next(iter(calibration.values()))[0]["cdf_scores"])

    # First, find the CDF of each calibration task's estimate quantile.
    trials = []
    cdfs = []
    p_deltas = []
    n_deltas = []
    for task, few_shot_trials in calibration.items():
        trial_idx = np.random.choice(len(few_shot_trials))
        trials.append((task, trial_idx))
        trial = few_shot_trials[trial_idx]
        pred_quantile = trial["quantile"]
        rank = (trial["cdf_scores"] <= pred_quantile).sum()
        for i in range(rank, len(trial["cdf_scores"])):
            p_deltas.append(trial["cdf_scores"][i] - trial["cdf_scores"][rank - 1])
        for i in range(rank - 1, -1, -1):
            n_deltas.append(trial["cdf_scores"][rank - 1] - trial["cdf_scores"][i])
        cdfs.append(rank / len(trial["cdf_scores"]))

    # Next, find the value of tau that completes the residual.
    residual = target - 1 / (n + 1) * sum(cdfs)
    if residual == 0:
        tau = 0
    else:
        if residual > 0:
            idx = math.ceil(residual * m * (n + 1))
            deltas = sorted(p_deltas)
        elif residual < 0:
            idx = math.floor(-residual * m * (n + 1))
            deltas = [-d for d in sorted(n_deltas)]
        if idx < len(deltas):
            tau = deltas[idx]
        else:
            tau = float("inf")

    # Verify that the sum is >= target.
    inflated_cdf = []
    for task, task_idx in trials:
        trial = calibration[task][task_idx]
        pred_quantile = trial["quantile"] + tau
        rank = (trial["cdf_scores"] <= pred_quantile).sum()
        inflated_cdf.append(rank / len(trial["cdf_scores"]))
    assert(sum(inflated_cdf) / (n + 1) >= target)

    # Next, iterate through the test data, using the inflated quantile.
    efficiencies = []
    accuracies = []

    # Run through each of the meta-tasks.
    for task, few_shot_trials in test.items():

        # Store task accuracies and efficienciess.
        task_accuracy = []
        task_efficiency = []

        # For each meta-task, evaluate each of the iterators of few-shot prediction.
        for trial in few_shot_trials:

            # Store trial accuracies and efficiencies.
            accuracy = []
            efficiency = []

            # Each few-shot iteration is evaluated over multiple query samples.
            for i in range(len(trial["output_scores"])):
                label = trial["output_labels"][i]
                quantile = trial["quantile"]
                prediction = []

                # Check each possible label.
                for y in [0, 1]:
                    score = trial["output_scores"][i][y]
                    if score <= quantile + tau:
                        prediction.append(y)

                # Compute accuracy and efficiency.
                accuracy.append(float(label in prediction))
                efficiency.append(len(prediction))

            # Update task metrics.
            task_accuracy.append(np.mean(accuracy))
            task_efficiency.append(np.mean(efficiency))

        # Update meta metrics.
        accuracies.append(np.mean(task_accuracy))
        efficiencies.append(np.mean(task_efficiency))

    accuracy_stats = stats(accuracies)
    efficiency_stats = stats(efficiencies)

    return dict(accuracy=accuracy_stats, efficiency=efficiency_stats)


def main(args):
    quantile = 1 - args.epsilon
    # quantile = np.sqrt(quantile)

    data = load(args.data, quantile)
    outputs = []
    num_calibration = int(args.p_calibration * len(data))
    for _ in tqdm.tqdm(range(args.num_trials), desc="evaluating trials"):
        tasks = list(data.keys())
        np.random.shuffle(tasks)
        calibration = {k: data[k] for k in tasks[:num_calibration]}
        test = {k: data[k] for k in tasks[num_calibration:]}
        meta = compute_meta_conformal(calibration, test, 1 - quantile)
        baseline = compute_baseline(calibration, 1 - quantile, use_correction=True)
        # meta_conservative = compute_meta_conformal_conservative(calibration, test, args.epsilon, bootstrap=False)
        outputs.append((meta, baseline))

    print(quantile)
    print("META:")
    print(np.mean([o[0]["accuracy"]["avg"] for o in outputs]))
    print(np.mean([o[0]["accuracy"]["std"] for o in outputs]))
    print(np.mean([o[0]["efficiency"]["avg"] for o in outputs]))
    print(np.mean([o[0]["efficiency"]["std"] for o in outputs]))
    print(" ")
    print("Baseline:")
    print(np.mean([o[1]["accuracy"]["avg"] for o in outputs]))
    print(np.mean([o[1]["accuracy"]["std"] for o in outputs]))
    print(np.mean([o[1]["efficiency"]["avg"] for o in outputs]))
    print(np.mean([o[1]["efficiency"]["std"] for o in outputs]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--p_calibration", type=float, default=0.9)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--use_correction", action="store_true")
    args = parser.parse_args()
    main(args)