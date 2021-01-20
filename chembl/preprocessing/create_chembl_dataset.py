"""Prepare ChEMBL dataset for combinatoric property prediction.

1) Filter valid smiles.
2) Filter properties that have low variance/few molecules.
3) Split molecules randomly to train/dev/test.
4) Write CSV files.
"""

import argparse
import collections
import csv
import numpy as np
import os
import subprocess
import tqdm
from rdkit import Chem

CHEMBL_PATH = "/data/rsg/chemistry/swansonk/antibiotic_moa/data/pchembl_100.csv"

Molecule = collections.namedtuple(
    "Molecule", ["smiles", "targets"])

MoleculeProperty = collections.namedtuple(
    "Property", ["smiles", "value"])


def filter_invalid_smiles(smiles):
    if not smiles:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumHeavyAtoms() == 0:
        return True
    return False


def load_dataset(path):
    """Return list of molecules --> attributes."""
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        smiles_column = columns[0]
        target_columns = columns[1:]

        # Read in all the dataset smiles.
        dataset = []
        num_lines = int(subprocess.check_output(["wc", "-l", path], encoding="utf8").split()[0])
        for row in tqdm.tqdm(reader, total=num_lines, desc="reading smiles"):
            smiles = row[smiles_column]
            if filter_invalid_smiles(smiles):
                continue
            datapoint = Molecule(smiles, {t: float(row[t]) for t in target_columns if row[t]})
            dataset.append(datapoint)

        return dataset


def group_targets(dataset, args):
    """Map targets to molecules."""
    targets_to_molecules = collections.defaultdict(list)
    for mol in dataset:
        for target, value in mol.targets.items():
            targets_to_molecules[target].append(
                MoleculeProperty(mol.smiles, value))

    filtered = {}
    for target, molecules in targets_to_molecules.items():
        if len(molecules) < args.min_molecules_per_target:
            continue
        if np.std([mol.value for mol in molecules]) < args.min_std_per_target:
            continue
        indices = np.random.permutation(len(molecules))[:args.max_molecules_per_target]
        filtered[target] = [molecules[i] for i in indices]

    return filtered


def split_targets(dataset, args):
    """Partition targets into splits."""
    val_size = int(args.val_p * len(dataset))
    test_size = int(args.test_p * len(dataset))
    train_size = len(dataset) - val_size - test_size

    targets = list(dataset.keys())
    np.random.shuffle(targets)
    train_targets = targets[:train_size]
    val_targets = targets[train_size:train_size + val_size]
    test_targets = targets[train_size + val_size:]

    train = {t: dataset[t] for t in train_targets}
    val = {t: dataset[t] for t in val_targets}
    test = {t: dataset[t] for t in test_targets}

    print("=" * 50)
    print("Num train targets: %d" % len(train))
    print("Num val targets: %d" % len(val))
    print("Num test targets: %d" % len(test))
    print("=" * 50)

    return [train, val, test]


def split_train(dataset, args):
    """Partition into folds."""
    indices = np.arange(len(dataset))
    fold_indices = np.array_split(indices, args.train_folds)
    targets = list(dataset.keys())

    folds = [dict(train={}, val={}) for _ in range(args.train_folds)]
    for i in range(args.train_folds):
        for idx in fold_indices[i]:
            target = targets[idx]
            folds[i]["val"][target] = dataset[target]
        for j in range(args.train_folds):
            if i == j:
                continue
            for idx in fold_indices[j]:
                target = targets[idx]
                folds[i]["train"][target] = dataset[target]

    return folds


def main(args):
    np.random.seed(args.seed)

    # Load chembl dataset.
    dataset = load_dataset(args.dataset_file)

    # Group by targets.
    dataset = group_targets(dataset, args)

    # Split targets into train/val/test.
    splits = split_targets(dataset, args)

    # Split train into K folds.
    train_folds = split_train(splits[0], args)

    # Name splits.
    names = ["train", "val", "test"]
    for i, fold in enumerate(train_folds):
        splits.extend([fold["train"], fold["val"]])
        names.extend(["train_f%d" % i, "val_f%d" % i])

    # Write splits.
    os.makedirs(args.output_dir, exist_ok=True)
    for i, name in enumerate(names):
        molecule_file = os.path.join(args.output_dir, "%s_molecules.csv" % name)
        with open(molecule_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "target", "value"])
            for target, molecules in splits[i].items():
                for molecule in molecules:
                    row = [molecule.smiles, target, str(molecule.value)]
                    writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_folds", type=int, default=5)
    parser.add_argument("--val_p", type=float, default=0.15)
    parser.add_argument("--test_p", type=float, default=0.15)
    parser.add_argument("--min_std_per_target", type=float, default=0.5)
    parser.add_argument("--min_molecules_per_target", type=int, default=300)
    parser.add_argument("--max_molecules_per_target", type=int, default=1000)
    parser.add_argument("--dataset_file", default=CHEMBL_PATH, type=str,
                        help="Path to ChEMBL dataset.")
    parser.add_argument("--output_dir", default="../data/chembl", type=str,
                        help="Path to directory where tasks will be written.")
    parser.add_argument("--seed", default=42, type=int,
                        help="Random seed for reproducibility.")
    main(parser.parse_args())
