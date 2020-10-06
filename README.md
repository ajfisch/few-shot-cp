# Meta-learned Few-Shot Conformal Prediction


## Task Instructions

1. Train a nonconformity measure using a standard meta-learned few-shot learning algorithm.

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
    ...
```


2. Train quantiles. Train using nonconformity scores measure on K - 1 support labels. Potentially add encodings of inputs as well (like context augmentation in adaptive risk minimization). Train for 0.95/0.9/0.8/0.7/0.6 quantiles.

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

In order to train for the epsilon-quantile overall, we need to optimize alpha * beta = epsilon. For conformalization on the task level, beta can only go as high as 1 - (1 / num_calibration_tasks). Generally will want to hyper-parameter sweep 1 - (i / num_calibration_tasks) for i in [1, 2, 3].

3. Compute nonconformity scores + quantiles on test data. Apply conformal prediction.
