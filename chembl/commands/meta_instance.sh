#! /bin/bash

K=16
set -ex

for q in 95 90 80 70 60; do
    python meta_instance_cp.py \
           --num_trials 2000 \
           --dataset_file "../ckpts/chembl/k=$K/conformal/q=0.$q/val.jsonl" \
           --tolerances 0.$q \
           --deltas 0 0.9 \
           --overwrite_results
done
