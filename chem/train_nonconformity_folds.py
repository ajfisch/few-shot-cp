#! /bin/python
"""Output commandline args for training folds."""

for k in [2, 4, 8, 16, 32]:
    for fold in range(10):
        print("python few_shot.py " +
              "--default_root_dir \"../ckpts/chembl/few_shot/random/k=%d/fold_%d\" " % (k, fold) +
              "--train_data \"../data/chembl/random/train_folds/fold_[!%d]/*.csv\" " % (fold) +
              "--train_features \"../data/chembl/random/train_folds/fold_[!%d]/*.npy\" " % (fold) +
              "--num_support %d " % k +
              "--gpus 1")
    print("python few_shot.py " +
          "--default_root_dir \"../ckpts/chembl/few_shot/random/k=%d/all_folds\" " % (k) +
          "--num_support %d " % k +
          "--gpus 1")
