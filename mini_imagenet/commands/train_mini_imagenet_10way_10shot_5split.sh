#!/usr/bin/env bash

set -ex

# Split dataset 5 ways for cross-val
python prototypical_networks/data_split.py

# Train on all data.
python prototypical_networks/train_mini_imagenet.py \
    --arch default_convnet \
    --epochs 200 \
    --print-freq 20 \
    --optimizer adam \
    --step_size 20 \
    --gamma 0.5 \
    --lr 0.001 \
    -j 24 \
    --n_query_train 15 \
    --n_query_val 15 \
    --n_support 10 \
    --n_way_train 10 \
    --n_way_val 10 \
    --n_episodes_train 100 \
    --n_episodes_val 400 \
    --model_name mini_imagenet_10_shot_10_way

# Train cross-val models.
for split in 0 1 2 3 4; do
    python prototypical_networks/train_mini_imagenet.py \
    --splits_path mini_imagenet_5_splits/$split/ \
    --arch default_convnet \
    --epochs 200 \
    --print-freq 20 \
    --optimizer adam \
    --step_size 20 \
    --gamma 0.5 \
    --lr 0.001 \
    -j 24 \
    --n_query_train 15 \
    --n_query_val 15 \
    --n_support 10 \
    --n_way_train 10 \
    --n_way_val 10 \
    --n_episodes_train 100 \
    --n_episodes_val 400 \
    --model_name mini_imagenet_10_shot_10_way_split_$split
done
