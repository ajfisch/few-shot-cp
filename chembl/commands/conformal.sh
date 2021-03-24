#! /bin/bash

K=16
set -ex

for q in 95 90 80 70 60; do
    python preprocessing/create_conformal_dataset.py \
         --q_ckpt "../ckpts/chembl/k=$K/quantile/q=0.$q/model.pt" \
         --s_ckpt "../ckpts/chembl/k=$K/nonconformity/full/model.pt" \
         --tolerance 0.$q \
         --output_file "../ckpts/chembl/k=$K/conformal/q=0.$q/test.jsonl"
done
