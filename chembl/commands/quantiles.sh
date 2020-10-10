#! /bin/bash

for quantile in 0.95 0.90 0.80 0.70 0.60; do
    echo python quantile_snn.py \
         --checkpoint_dir "../ckpts/chembl/k=10/quantile_snn/q=${quantile}" \
         --num_data_workers 5 \
         --alpha $quantile \
         --use_adaptive_encoding false
done
