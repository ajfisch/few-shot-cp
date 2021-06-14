# FewRel classification Experiments

The classifier here is based on Prototypical Networks for text classification, as implemented by https://github.com/thunlp/FewRel

## Environment Preparations
To load the prototypical networks code run:
```
git submodule init
git submodule update
```

## Preprocessing

Download [FewRel 1.0](https://thunlp.github.io/1/fewrel1.html) training and validation data and place it in folder `data/'


## Train Nonconformity Measure

`bash commands/train_fewrel_10_way_10_shot_5split.sh`

Creates nonconformity measure over with CNN encoders based on distance to prototypes. Follows a 10-way 16-shot setting for training the nonconformity score.

`python create_quantile_dataset.py`

Create dataset for training the quantile predictor by evaluating on multiple folds. Aggregate predictions.

Data will be saved to `results/`


## Train Quantile Predictor

`python quantile_snn.py --alpha 0.80 --checkpoint_dir results/10_way/quantile_snn/q=0.80`

Create quantile predictor for a level alpha=`0.80`.

`python create_conformal_dataset.py --q_ckpt results/10_way/quantile_snn/q=0.80\/weights.pt.ckpt --alpha 0.80  --n_support 16 --output_file results/10_way/quantile_snn/q=0.80\/conf_val.jsonl
`

Dump all outputs for conformal processing


## Compute Conformal Results

`python meta_cp.py`
