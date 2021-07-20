# ChEMBL Regression Experiments


Every data point that has a pChEMBL value (i.e., a normalized log activity value, see https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions#what-is-pchembl). Initial preprocessing from the ChEMBL dataset averaged duplicates and filters to only include assays with at least 100 molecules.


## Preprocessing

`python preprocessing/create_chembl_dataset.py`

Creates chembl dataset folds. Molecules are grouped by assay and filtered to have [min, max] number of molecules, with sufficient variance in pchembl scores.

`python preprocessing/save_features.py`

Creates chembl rdkit features for molecules.

Outputs are saved to `../data/chembl`.

Our preprocessed folds are available for download [here](https://people.csail.mit.edu/fisch/assets/data/few-shot-cp/chembl.tar.gz).

## Train Nonconformity Measure

`python modeling/nonconformity.py`

Creates nonconformity measure over chembl data. This is a few-shot ridge regressor.

`python preprocessing/create_quantile_dataset.py`

Create dataset for training the quantile predictor by evaluating on multiple folds. Aggregate predictions.


## Train Quantile Predictor

`python modeling/quantile.py`

Create quantile predictor for a level alpha.

`python preprocessing/create_conformal_dataset.py`

Dump all outputs for conformal processing.


## Compute Conformal Results

`python meta_cp.py`
