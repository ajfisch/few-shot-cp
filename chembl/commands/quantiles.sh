#! /bin/bash
K=16

for quantile in 0.05; do
    echo python modeling/quantile.py \
         --checkpoint_dir "../ckpts/chembl/k=$K/quantile/q=${quantile}" \
         --train_data "../ckpts/chembl/k=$K/nonconformity/train_quantile.jsonl" \
         --val_data "../ckpts/chembl/k=$K/nonconformity/val_quantile.jsonl" \
         --num_data_workers 5 \
         --alpha $quantile \
         --use_adaptive_encoding false
done
