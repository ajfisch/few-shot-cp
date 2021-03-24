#! /bin/bash
K=16

for num_support in 128 256 512; do
    echo python preprocessing/create_quantile_dataset.py \
         --num_support $num_support \
         --num_data_workers 5 \
         --output_dir "../ckpts/chembl/k=${num_support}/nonconformity"
done
