#! /bin/bash

set -ex

K=16

declare -A epsilons=(
    ["0.95"]="0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99"
    ["0.90"]="0.85 0.86 0.87 0.88 0.89 0.90 0.91 0.92 0.93 0.94 0.95"
    ["0.80"]="0.75 0.76 0.77 0.78 0.79 0.80 0.81 0.82 0.83 0.84 0.85"
    ["0.70"]="0.65 0.66 0.66 0.68 0.69 0.70 0.71 0.72 0.73 0.74 0.75"
    ["0.60"]="0.55 0.56 0.57 0.58 0.59 0.60 0.61 0.62 0.63 0.64 0.65"
)

for q in 95 90 80 70 60; do
    python meta_instance_cp.py \
           --num_trials 5000 \
           --dataset_file "../ckpts/chembl/k=$K/conformal/q=0.$q/val.jsonl" \
           --deltas 0 0.9 \
           --tolerances ${epsilons["0.$q"]} \
           --overwrite_results
done
