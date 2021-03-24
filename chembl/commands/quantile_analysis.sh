#! /bin/bash

set -ex
q=80

for K in 1 2 4 8 16 32 64 128 256 512; do
    python preprocessing/quantile_predictions.py \
         --q_ckpt "../ckpts/chembl/k=$K/quantile/q=0.$q/model.pt" \
         --s_ckpt "../ckpts/chembl/k=16/nonconformity/full/model.pt" \
         --tolerance 0.$q \
         --num_support $K \
         --output_file "../ckpts/chembl/k=$K/conformal/q=0.$q/quantile_analysis.jsonl"
done
