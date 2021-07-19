# ChEMBL Regression Experiments


Every data point that has a pChEMBL value (i.e., a normalized log activity value, see https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions#what-is-pchembl), and then it averages duplicates and filters to only include assays with at least X number of molecules).

Different variations:

```
Minimum number of molecules per assay = 100
Number of assays = 1,499
Number of molecules = 1,661,044
Path = /data/rsg/chemistry/swansonk/antibiotic_moa/data/pchembl_100.csv
```

```
Minimum number of molecules per assay = 1000
Number of assays = 120
Number of molecules = 1,336,684
Path = /data/rsg/chemistry/swansonk/antibiotic_moa/data/pchembl_1000.csv
```

```
Minimum number of molecules per assay = 10000
Number of assays = 38
Number of molecules = 1,058,663
Path = /data/rsg/chemistry/swansonk/antibiotic_moa/data/pchembl_10000.csv
```

## Preprocessing

`python preprocessing/create_chembl_dataset.py`

Creates chembl dataset folds. Molecules are grouped by assay and filtered to have [min, max] number of molecules, with sufficient variance in pchembl scores.

`python preprocessing/save_features.py`

Creates chembl rdkit features for molecules.

Outputs are saved to `../data/chembl`.

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
