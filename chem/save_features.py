"""Prepare RDKit features for chemprop models."""

import argparse
import functools
import glob
import multiprocessing
import os
import tqdm
import numpy as np
import chemprop


def generate_features_for_file(paths, features_generator):
    smiles_path, features_path = paths

    smiles_to_idx = {}
    idx_to_features = []

    features_generator = chemprop.features.get_features_generator(features_generator)
    for i, smiles in enumerate(chemprop.data.get_smiles(smiles_path)):
        smiles_to_idx[smiles] = i
        features = np.array(features_generator(smiles))
        features = np.where(np.isnan(features), 0, features)
        idx_to_features.append(features)

    idx_to_features = np.stack(idx_to_features)
    np.save(features_path, (smiles_to_idx, idx_to_features))


def main(args):
    smiles_paths = glob.glob(args.file_pattern)
    features_paths = []
    for smiles_file in smiles_paths:
        basename = os.path.splitext(os.path.basename(smiles_file))[0]
        features_dir = args.features_dir
        if features_dir is None:
            features_dir = os.path.dirname(smiles_file)
        features_paths.append(os.path.join(features_dir, basename + ".npy"))
    paths = list(zip(smiles_paths, features_paths))
    num_workers = min(args.num_threads, len(paths))
    workers = multiprocessing.Pool(num_workers)
    worker_fn = functools.partial(generate_features_for_file, features_generator=args.features_generator)
    with tqdm.tqdm(total=len(paths), desc="computing and saving features") as pbar:
        for _ in workers.imap_unordered(worker_fn, paths):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument("--file_pattern", type=str, default="../data/chembl/scaffold/*/*.csv")
    parser.add_argument("--features_dir", type=str, default=None)
    parser.add_argument("--features_generator", type=str, default="rdkit_2d_normalized")
    main(parser.parse_args())
