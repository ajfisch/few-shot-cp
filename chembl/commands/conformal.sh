#! /bin/bash


echo python create_conformal_dataset.py \
     --q_ckpt "../ckpts/chembl/k=10/quantile_snn/q=0.95/_ckpt_epoch_11.ckpt" \
     --s_ckpt "../ckpts/chembl/k=10/conformal_mpn/full/_ckpt_epoch_11.ckpt" \
     --epsilon 0.04 \
     --output_file "../ckpts/chembl/k=10/conformal/q=0.96/val.jsonl"


echo python create_conformal_dataset.py \
     --q_ckpt "../ckpts/chembl/k=10/quantile_snn/q=0.90/_ckpt_epoch_11.ckpt" \
     --s_ckpt "../ckpts/chembl/k=10/conformal_mpn/full/_ckpt_epoch_11.ckpt" \
     --epsilon 0.09 \
     --output_file "../ckpts/chembl/k=10/conformal/q=0.91/val.jsonl"


echo python create_conformal_dataset.py \
     --q_ckpt "../ckpts/chembl/k=10/quantile_snn/q=0.80/_ckpt_epoch_11.ckpt" \
     --s_ckpt "../ckpts/chembl/k=10/conformal_mpn/full/_ckpt_epoch_11.ckpt" \
     --epsilon 0.19 \
     --output_file "../ckpts/chembl/k=10/conformal/q=0.81/val.jsonl"


echo python create_conformal_dataset.py \
     --q_ckpt "../ckpts/chembl/k=10/quantile_snn/q=0.70/_ckpt_epoch_12.ckpt" \
     --s_ckpt "../ckpts/chembl/k=10/conformal_mpn/full/_ckpt_epoch_11.ckpt" \
     --epsilon 0.29 \
     --output_file "../ckpts/chembl/k=10/conformal/q=0.71/val.jsonl"


echo python create_conformal_dataset.py \
     --q_ckpt "../ckpts/chembl/k=10/quantile_snn/q=0.60/_ckpt_epoch_10.ckpt" \
     --s_ckpt "../ckpts/chembl/k=10/conformal_mpn/full/_ckpt_epoch_11.ckpt" \
     --epsilon 0.39 \
     --output_file "../ckpts/chembl/k=10/conformal/q=0.61/val.jsonl"
