#! /bin/python
"""Output commandline args for training folds."""
import math

for quantile in [0.95, 0.90, 0.80, 0.70]:
    for loss in ["quantile", "mse"]:
        print("python quantile_regression.py " +
              "--default_root_dir \"../ckpts/chembl/quantiles/random/q=%2.2f,l=%s,mol=false\" " % (quantile, loss) +
              "--encode_molecules false " +
              "--quantile %f " % math.sqrt(quantile) +
              "--loss %s " % loss +
              "--gpus 1")
