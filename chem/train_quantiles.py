#! /bin/python
"""Output commandline args for training folds."""
import math

for quantile in [0.95, 0.90, 0.80, 0.70]:
    print("python quantile_regression.py " +
          "--default_root_dir \"../ckpts/chembl/quantiles/random/k=10/q=%2.2f,mol=false\" " % (quantile) +
          "--encode_molecules false " +
          "--quantile %f " % quantile +
          "--loss mse " +
          "--gpus 1")
