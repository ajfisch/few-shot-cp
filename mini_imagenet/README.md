# Mini ImageNet classification Experiments

The classifier here is based on Prototypical Networks, as implemented by https://github.com/jsalbert/prototypical-networks.

## Environment Preparations
To load the prototypical networks code run:
```
git submodule init
git submodule update
```

## Preprocessing

Download [Mini-ImageNet images]( https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and place them on folder `mini_imagenet/'


## Train Nonconformity Measure

`bash commands/train_mini_imagenet_10way_10shot_5split.sh`

Creates nonconformity measure over Mini ImageNet based on distance to prototypes. Follows a 10-way 10-shot setting for training the nonconformity score.

`python create_quantile_dataset.py`

Create dataset for training the quantile predictor by evaluating on multiple folds. Aggregate predictions.

Data will be saved to `results/10_way`


## Train Quantile Predictor

`python quantile_snn.py --alpha 0.80 --checkpoint_dir results/10_way/quantile_snn/q=0.80`

Create quantile predictor for a level alpha=`0.80`.

`python create_conformal_dataset.py --q_ckpt results/10_way/quantile_snn/q=0.80\/weights.pt.ckpt --alpha 0.80  --n_support 16 --output_file results/10_way/quantile_snn/q=0.80\/conf_val.jsonl
`

Dump all outputs for conformal processing. Here the setting is 10 way 16 shot (It can be different than the setting used to train the nonconformity measure).


## Compute Conformal Results

`python meta_cp.py`
