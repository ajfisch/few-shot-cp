#! /bin/python
"""Output commandline args for training folds."""

for fold in range(10):
    print("python few_shot.py " +
          "--default_root_dir \"../ckpts/chembl/few_shot/random/fold_%d\" " % fold +
          "--train_data \"../data/chembl/random/train_folds/fold_[!%d]/*.csv\" " % fold +
          "--train_features \"../data/chembl/random/train_folds/fold_[!%d]/*.npy\" " % fold +
          "--gpus 1")
