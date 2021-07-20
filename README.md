# Few-shot Conformal Prediction with Auxiliary Tasks

Code for [Few-shot Conformal Prediction with Auxiliary Tasks](https://arxiv.org/abs/2102.08898).

## Abstract

For many domains, pairing predictions with well-calibrated probabilities is important for quantifying uncertainty. For example, medical decisions would ideally be based on calculated probabilities that reflect reality. Unfortunately, for many of these tasks, in-domain data can be severely limited. Furthermore, ensuring calibrated probabilities for each possible outcome is hard; conformal prediction tackles the intermediate problem of assigning calibrated probabilities to sets of outcomes. In this project we are able to adapt conformal prediction to few-shot settings by casting it as a meta-learning paradigm over exchangeable collections of auxiliary tasks.

## Conformal Prediction
All of the code for analyzing cascaded conformal predictions is in the `meta_conformal` directory. Examples for how to call into it are given in the tasks subdirectories. Note that in this repository, the functions exposed in `meta_conformal` are for analysis only, i.e. they are not implemented as a real-time predictors.

Efficient implementations of online predictors for the tasks considered here might be included later.

## Modeling

Each tasks build upon existing meta-learning architectures for few-shot learning. The classification tasks (`fewrel`, `mini_imagenet`) both use [prototypical networks](https://arxiv.org/abs/1703.05175), while the regression task (`chembl`) uses the [R2D2 framework](https://openreview.net/forum?id=HyxnZh0ct7).

## Task Instructions

Each subdirectory contains instructions for their respective tasks. In general, for a new task, we use the following procedure:

### Meta nonconformity measures
Train a nonconformity measure using a standard meta-learned few-shot learning algorithm.

Example API is:

```python
def forward(self, query, support, support_targets):
    """Compute nonconformity scores (loss).

    Args:
        query: <float>[tasks_per_batch, n_query, dim]
               Encodings of test points for prediction.

        support: <float>[tasks_per_batch, n_support, dim]
               Encodings of support points for few-shot learning.

        support_targets: <T>[tasks_per_batch, n_support]
               Target labels for the support points.

    Returns:
        scores: <float>[tasks_per_batch, n_query]
               Nonconformity scores for the query points.
    """
```

The important thing is to return the nonconformity scores.

### Meta quantile predictors
Train quantile predictors. Train using nonconformity scores measure on K - 1 support labels (leave-out-one). One could also add encodings of inputs as well (like context augmentation in [adaptive risk minimization](https://arxiv.org/abs/2007.02931)). Train for 0.95/0.9/0.8/0.7/0.6 quantiles.

Example API:

```python
def forward(self, support_scores, support=None, query=None):
    """Estimate quantile.

    Args:
        support_scores: <float>[tasks_per_batch, n_support]
            Nonconformity scores computed on Kth support example using the other K-1 examples.

        support: <float>[tasks_per_batch, n_support, dim]
            Optional. Encodings of support points for few-shot learning.

        query: <float>[tasks_per_batch, n_query, dim]
            Optional. Encodings of query points for prediction.

    Returns:
        quantile: <float>[tasks_per_batch]
            Scalar estimate of task quantile given K examples.
    """
    ...
```

### Meta calibration
Compute calibration of nonconformity scores + quantiles on test data. This uses the `meta_conformal` prediction functions.

The important thing to note is that validity scores are computed based on marginalizing over task and task examples (for calibration/support vs test). For each task, we compute a batch of examples with different support/test splits and record their scores (see `create_conformal_dataset` in each of the subdirectories). After this is done, the main access point for calculating results is via importing:

```python
from meta_conformal import instance_cp
```

Then to evaluate, run:

```python
instance_cp.evaluate_trials(
    # List of (task id, episode id) from 1 to T + 1 and then randomly from 1 to m_i determining the order of tasks, 
    # and which episode (support/test scenario) to use for that task.
    trials=trials,

    # List of task (from 1 to T + 1) data, each entry being a list (batch) of dictionaries, each containing:
    #    pred_quantile: scalar quantile prediction.
    #    query_preds: list of n_calibration predictions (i.e., raw score).
    #    query_scores: list of n_calibration nonconformity scores (e.g., |target - raw score| for regression).
    #    query_targets: list of n_calibration prediction targets.
    #    support_encs: list of k support encodings (i.e., for RRCM baseline prediction).
    #    support_targets: list of k support prediction targets.
    #    query_encs: list of n_calibration prediction targets.
    #    ... (other fields possible, see examples) ...
    #    l2_lambda: Optional l2 parameter used for regression tasks (from R2D2).
    tasks=tasks,
    
    # 1 - \epsilon parameter for meta-cp.
    epsilon=epsilon,

    # 1 - \delta parameter for calibration conditional meta-cp.
    delta=delta,
    
    # Either regression or classification (for computing baselines).
    task_type="classification",
    
    # Number of threads for multiprocessing.
    threads=args.threads,
)
```

## Citation

If you use this in your work please cite:

```
@inproceedings{fisch2021fewshot,
    title={Few-shot Conformal Prediction with Auxiliary Tasks},
    author={Adam Fisch and Tal Schuster and Tommi Jaakkola and Regina Barzilay},
    booktitle={Proceedings of The Thirty-eighth International Conference on Machine Learning},
    year={2021},
}
```
    
