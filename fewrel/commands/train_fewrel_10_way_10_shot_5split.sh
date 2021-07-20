#!/usr/bin/env bash

set -ex

# Split dataset 5 ways for cross-val
python FewRel/data_split.py

# Train on all data.
python FewRel/train_demo.py \
--train train_wiki \
--val val_wiki \
--test val_wiki \
--trainN 10 \
--N 10 \
--K 10 \
--Q 5 \
--model proto \
--encoder cnn \
--val_step 1000 \
--batch_size 32

# Train cross-val models.
for split in 0 1 2 3 4; do
    python FewRel/train_demo.py \
    --train wiki_5_splits/$split\/train \
    --val wiki_5_splits/$split\/val \
    --test val_wiki \
    --ckpt_name $split \
    --trainN 10 \
    --N 10 \
    --K 10 \
    --Q 5 \
    --model proto \
    --encoder cnn \
    --val_step 1000 \
    --batch_size 32
done
