#! /bin/bash
K=16

echo python modeling/nonconformity.py --gpus 0 --checkpoint_dir="../ckpts/chembl/k=$K/nonconformity/full" --num_support $K
for fold in 0 1 2 3 4; do
    echo python modeling/nonconformity.py \
         --gpus 0 \
         --checkpoint_dir="../ckpts/chembl/k=$K/nonconformity/$fold" \
         --num_support $K \
         --train_data="../data/chembl/train_f${fold}_molecules.csv" \
         --train_features="../data/chembl/features/train_f${fold}_molecules.npy"
done
