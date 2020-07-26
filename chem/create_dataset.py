"""Prepare ChEMBL dataset for few-shot setting.

1) Filter valid smiles.
2) Filter properties to those with sufficient examples.
3) Balance positive and negative samples.
4) Write out individual property CSVs.
"""

import argparse
import csv
import collections
import subprocess
import tqdm
import os

import chemprop
import numpy as np
import rdkit

CHEMBL_PATH = "/data/scratch/fisch/third_party/chemprop/data/chembl.csv"


def filter_invalid_smiles(smiles):
    if not smiles:
        return True
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol.GetNumHeavyAtoms() == 0:
        return True
    return False


def load_dataset(path, N):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        smiles_column = columns[0]
        target_columns = columns[1:]

        # Keep track of property --> list of molecules (by active/inactive).
        property_to_smiles = collections.defaultdict(lambda: collections.defaultdict(list))

        # Read in all the dataset smiles.
        num_lines = int(subprocess.check_output(["wc", "-l", path], encoding="utf8").split()[0])
        for row in tqdm.tqdm(reader, total=num_lines, desc="reading smiles"):
            smiles = row[smiles_column]
            if filter_invalid_smiles(smiles):
                continue
            scaffold = chemprop.data.scaffold.generate_scaffold(smiles)
            for target in target_columns:
                value = row[target]
                if not value:
                    continue
                value = int(value)
                property_to_smiles[target][value].append((scaffold, smiles))

        # Filter properties with sufficient examples.
        valid_properties = {}
        for target, values in property_to_smiles.items():
            if len(set([scaffold for scaffold, _ in values[0]])) < N:
                continue
            if len(set([scaffold for scaffold, _ in values[1]])) < N:
                continue
            valid_properties[target] = values

        print("Initially kept %d of %d properties." % (len(valid_properties), len(property_to_smiles)))
        return valid_properties


def make_splits(dataset, num_test, num_val, num_train, N):
    targets = list(dataset.keys())
    np.random.shuffle(targets)

    def _choose(values, exclude_scaffolds):
        # Shuffle values.
        np.random.shuffle(values)

        # Filter to disjoint scaffolds.
        scaffolds = set()
        filtered = []
        for scaffold, smiles in values:
            if scaffold not in scaffolds and scaffold not in exclude_scaffolds:
                filtered.append((scaffold, smiles))
                scaffolds.add(scaffold)

        # Sample N values.
        if len(filtered) < N:
            return None
        samples = np.random.choice(len(filtered), N, replace=False)
        return [filtered[i] for i in samples]

    def _update_scaffolds(split, exclude_scaffolds):
        for target in split:
            for active in [0, 1]:
                for scaffold, _ in dataset[target][active]:
                    exclude_scaffolds.add(scaffold)

    # Filter overlaps.
    used = set()

    # Gather test properties.
    test = []
    filtered = 0
    for i, target in enumerate(targets):
        if len(test) == num_test:
            break
        include = True
        for active in [0, 1]:
            samples = _choose(dataset[target][active], used)
            if not samples:
                include = False
                continue
            dataset[target][active] = samples
        if include:
            test.append(target)
        else:
            filtered += 1

    print("Filtered %d overlaps from test." % filtered)
    targets = targets[i:]
    _update_scaffolds(test, used)

    # Gather val properties, without molecule overlap in test.
    val = []
    filtered = 0
    for i, target in enumerate(targets):
        if len(val) == num_val:
            break
        include = True
        for active in [0, 1]:
            samples = _choose(dataset[target][active], used)
            if not samples:
                include = False
                continue
            dataset[target][active] = samples
        if include:
            val.append(target)
        else:
            filtered += 1

    print("Filtered %d overlaps from val." % filtered)
    targets = targets[i:]
    _update_scaffolds(val, used)

    # Gather train properties, without molecule overlap in val/test.
    train = []
    filtered = 0
    for target in targets:
        if len(train) == num_train:
            break
        include = True
        for active in [0, 1]:
            samples = _choose(dataset[target][active], used)
            if not samples:
                include = False
                continue
            dataset[target][active] = samples
        if include:
            train.append(target)
        else:
            filtered += 1

    print("Filtered %d overlaps from train." % filtered)

    splits = {"val": val, "test": test, "train": train}
    for split, keys in splits.items():
        print("%s: %d" % (split, len(keys)))

    return splits


def main(args):
    np.random.seed(args.seed)
    dataset = load_dataset(args.dataset_file, args.N)
    splits = make_splits(dataset, args.test, args.val, args.train, args.N)
    for split, keys in splits.items():
        output_dir = os.path.join(args.output_dir, split)
        os.makedirs(output_dir, exist_ok=True)
        split_dataset = {}
        for target in keys:
            rows = []
            for active in [0, 1]:
                rows.extend([[smiles, active] for _, smiles in dataset[target][active]])
            split_dataset[target] = rows
        for target, rows in tqdm.tqdm(split_dataset.items(), desc="writing %s files" % split):
            filename = os.path.join(output_dir, target + ".csv")
            with open(filename, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["smiles", target])
                for row in rows:
                    writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", default=200, type=int,
                        help="Minimum number of examples per class.")
    parser.add_argument("--train", default=100, type=int,
                        help="Number of tasks for training.")
    parser.add_argument("--val", default=25, type=int,
                        help="Number of tasks to hold-out for validation.")
    parser.add_argument("--test", default=25, type=int,
                        help="Number of tasks to hold-out for testing.")
    parser.add_argument("--dataset_file", default=CHEMBL_PATH, type=str,
                        help="Path to ChEMBL dataset.")
    parser.add_argument("--output_dir", default="../data/chembl/scaffold", type=str,
                        help="Path to directory where tasks will be written.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducibility.")

    main(parser.parse_args())
