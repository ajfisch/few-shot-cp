#! /bin/bash

set -ex

python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.95,l=quantile,mol=false/lightning_logs/version_0/checkpoints/epoch=18.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.95,l=quantile,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.95,l=mse,mol=false/lightning_logs/version_0/checkpoints/epoch=9.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.95,l=mse,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.90,l=quantile,mol=false/lightning_logs/version_0/checkpoints/epoch=18.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.90,l=quantile,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.90,l=mse,mol=false/lightning_logs/version_0/checkpoints/epoch=18.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.90,l=mse,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.80,l=quantile,mol=false/lightning_logs/version_0/checkpoints/epoch=14.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.80,l=quantile,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.80,l=mse,mol=false/lightning_logs/version_0/checkpoints/epoch=9.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.80,l=mse,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.70,l=quantile,mol=false/lightning_logs/version_0/checkpoints/epoch=5.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.70,l=quantile,mol=false
python create_cp_data.py --ckpt ../ckpts/chembl/quantiles/random/q=0.70,l=mse,mol=false/lightning_logs/version_0/checkpoints/epoch=14.ckpt --output_dir ../data/chembl/random/conformal/proto/val/k=10/q=0.70,l=mse,mol=false
