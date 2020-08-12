#! /bin/bash

python cp_experiment.py --data ../data/chembl/random/conformal/proto/val/k=10/q=0.95,l=mse,mol=false/data.json --epsilon 0.05
python cp_experiment.py --data ../data/chembl/random/conformal/proto/val/k=10/q=0.90,l=mse,mol=false/data.json --epsilon 0.1
python cp_experiment.py --data ../data/chembl/random/conformal/proto/val/k=10/q=0.80,l=mse,mol=false/data.json --epsilon 0.2
python cp_experiment.py --data ../data/chembl/random/conformal/proto/val/k=10/q=0.70,l=mse,mol=false/data.json --epsilon 0.3
