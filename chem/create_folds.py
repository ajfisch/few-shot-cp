"""Split training set into folds."""

import argparse
import glob
import os
import numpy as np
import shutil


def main(args):
    smiles_paths = glob.glob(args.file_pattern)
    fold_idx = np.array_split(np.random.permutation(len(smiles_paths)), args.folds)
    for i, fold in enumerate(fold_idx):
        fold_dir = os.path.join(args.output_dir, "fold_%d" % i)
        os.makedirs(fold_dir, exist_ok=True)
        for idx in fold:
            smiles_file = smiles_paths[idx]
            basename = os.path.splitext(os.path.basename(smiles_file))[0]
            for ext in [".csv", ".npy"]:
                shutil.copy(os.path.splitext(smiles_file)[0] + ext,
                            os.path.join(fold_dir, basename) + ext)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_pattern", type=str, default="../data/chembl/random/train/*.csv")
    parser.add_argument("--output_dir", type=str, default="../data/chembl/random/train_folds")
    parser.add_argument("--folds", type=int, default=10)
    main(parser.parse_args())
