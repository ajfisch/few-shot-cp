#! /bin/bash
K=16

for K in 64 128 256 512; do
    echo python modeling/quantile.py \
         --checkpoint_dir "../ckpts/chembl/k=$K/quantile/q=0.80" \
         --train_data "../ckpts/chembl/k=$K/nonconformity/train_quantile.jsonl" \
         --val_data "../ckpts/chembl/k=$K/nonconformity/val_quantile.jsonl" \
         --num_data_workers 5 \
         --alpha 0.80 \
         --use_adaptive_encoding false
done
