"""Run the conformal prediction experiments."""

import argparse
import collections
import json
import numpy as np
import os

from meta_conformal import utils
from meta_conformal import instance_cp


def load_data(dataset_file):
    """Load data and sample into trials."""
    task2idx = {}
    idx2examples = collections.defaultdict(list)

    # Read examples and partition to tasks.
    with open(dataset_file, "r") as f:
        for line in f:
            example = json.loads(line)
            for key, value in example.items():
                if isinstance(value, list):
                    example[key] = np.array(value)
            if example["task"] not in task2idx:
                task2idx[example["task"]] = len(task2idx)
            idx = task2idx[example["task"]]
            idx2examples[idx].append(example)
    tasks = [idx2examples[i] for i in range(len(idx2examples))]

    num_tasks = len(task2idx)
    print("Number of tasks: %d" % num_tasks)

    episodes_per_task = np.mean([len(v) for v in tasks])
    print("Avg number of episodes per task: %2.2f" % episodes_per_task)

    return tasks


def create_trials(tasks, num_trials):
    """Create random sample of task order + k-shot."""
    trials = []
    for _ in range(num_trials):
        task_idx = []
        for i in np.random.permutation(len(tasks)).tolist():
            j = np.random.randint(len(tasks[i]))
            task_idx.append((i, j))
        trials.append(task_idx)
    return trials


def main(args):
    np.random.seed(42)

    args.epsilons = [1 - t for t in args.tolerances]

    print("Loading data...")
    tasks = load_data(args.dataset_file)

    if not args.trials_file:
        args.trials_file = os.path.splitext(args.dataset_file)[0]
        args.trials_file += ("-trials=%d.json" % args.num_trials)
        os.makedirs(os.path.dirname(args.trials_file), exist_ok=True)

    if not args.output_file:
        args.output_file = os.path.splitext(args.trials_file)[0]
        args.output_file += "-results.jsonl"
        os.makedirs(os.path.dirname(args.trials_file), exist_ok=True)

    if args.overwrite_results:
        open(args.output_file, "w").close()

    if os.path.exists(args.trials_file) and not args.overwrite_trials:
        print("Loading trials from %s" % args.trials_file)
        with open(args.trials_file, "r") as f:
            trials = json.load(f)
    else:
        trials = create_trials(tasks, args.num_trials)
        print("Writing trials to %s" % args.trials_file)
        with open(args.trials_file, "w") as f:
            json.dump(trials, f)

    print("Will write to %s" % args.output_file)
    for epsilon in args.epsilons:
        for delta in args.deltas:
            result = instance_cp.evaluate_trials(
                trials=trials,
                tasks=tasks,
                epsilon=epsilon,
                delta=delta,
                task_type="regression",
                threads=args.threads)
            utils.write_result(epsilon, delta, result, args.output_file, overwrite=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default="../ckpts/chembl/k=16/conformal/q=0.80/val.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--trials_file", type=str, default=None)
    parser.add_argument("--num_trials", type=int, default=2000)
    parser.add_argument("--overwrite_trials", action="store_true")
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument("--tolerances", nargs="+", type=float, default=[.80])
    parser.add_argument("--deltas", nargs="+", type=float, default=[0])
    parser.add_argument("--threads", type=int, default=30)
    args = parser.parse_args()
    main(args)
