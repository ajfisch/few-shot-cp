"""Prepare the input files for conformal prediction."""

import argparse
import json
import numpy as np
import os
import torch
import tqdm

import conformal_mpn
import exact_cp
import quantile_snn


def predict(s_model, q_model, inputs, args):
    """Predict quantiles and compute baselines for an input batch.

    Prediction follows steps:
       1) Compute leave-out-one scores on the support set.
       2) Predict the leave-out-none quantile.
       3) Compute the leave-out-none query scores.
       4) Compute the exact conformal prediction baseline.
       5) Compute the jacknife+ conformal prediction baseline.

    Args:
        s_model: ConformalMPN model.
        q_model: QuantileSNN model.
        inputs: Input batch of support and query examples.

    Returns:
        pred_quantiles: [batch_size]
        query_scores: [batch_size, n_query]
        query_targets: [batch_size, n_query]
        exact_intervals: [batch_size, n_query, 2]
        jk_pvalues: [batch_size, n_query]
    """
    batch_size = args.batch_size
    n_query = args.num_query
    n_support = s_model.hparams.num_support
    dim = s_model.hparams.enc_hidden_size

    inputs = s_model.transfer_batch_to_device(inputs, s_model.device)

    # Encode molecules.
    (query, support, support_targets), query_targets = s_model.encode(
        inputs, [batch_size, n_support, n_query])

    # --------------------------------------------------------------------------
    # Step 1.
    #
    # Compute leave-out-one scores on the support set.
    # These "scores" will be used as inputs to the quantile predictor.
    # --------------------------------------------------------------------------

    loo_queries = query.new(batch_size, n_support, 1, dim)
    loo_support = support.new(batch_size, n_support, n_support - 1, dim)
    loo_support_targets = support_targets.new(batch_size, n_support, n_support - 1)
    loo_query_targets = query_targets.new(batch_size, n_support, 1)

    for i in range(n_support):
        # All indices *except* this one for support.
        support_idx = [j for j in range(n_support) if j != i]
        support_idx = torch.LongTensor(support_idx).to(s_model.device)

        # Held out i-th support example is the "query".
        loo_queries[:, i] = support[:, i:i + 1, :].view(
            batch_size, 1, dim)

        # Remaining k-1 supporth examples are the "support".
        loo_support[:, i] = support.index_select(1, support_idx).view(
            batch_size, n_support - 1, dim)

        # "Support" targets.
        loo_support_targets[:, i] = support_targets.index_select(1, support_idx).view(
            batch_size, n_support - 1)

        # "Query" target.
        loo_query_targets[:, i] = support_targets[:, i:i + 1].view(
            batch_size, 1)

    # Reshape to parallel batch_size * n_support.
    # [batch_size * n_support, 1]
    loo_pred = s_model.head(
        loo_queries.view(batch_size * n_support, 1, dim),
        loo_support.view(batch_size * n_support, n_support - 1, dim),
        loo_support_targets.view(batch_size * n_support, n_support - 1))

    # [batch_size, n_support, 1]
    loo_pred = loo_pred.view(batch_size, n_support, 1)
    loo_scores = (loo_query_targets - loo_pred).pow(2)

    # [batch_size, n_support]
    loo_scores = loo_scores.view(batch_size, n_support)

    # --------------------------------------------------------------------------
    # Step 2.
    #
    # Using the leave-out-one scores on the support set, predict the alpha
    # quantile using the given quantile predictor model.
    #
    # We just save this prediction for now---we'll conformalize it later.
    # --------------------------------------------------------------------------

    # TODO: Allow for mol_graph, mol_features.
    # [batch_size]
    pred_quantiles = q_model.predict(scores=loo_scores)

    # --------------------------------------------------------------------------
    # Step 3.
    #
    # Using the nonconformity model with the *full* support set (all K examples),
    # predict nonconformity scores on the query points.
    #
    # Again, we just save this prediction for now. After the quantile above is
    # conformalized, we'll compare these scores to make predictions.
    # --------------------------------------------------------------------------

    # [batch_size, n_query]
    query_scores = s_model.head(query, support, support_targets)

    # --------------------------------------------------------------------------
    # Step 4.
    #
    # Exact conformal prediction baseline using RRCM. We use the *unconditional*
    # molecule embeddings from the nonconformity model as fixed input features
    # to the ridge regressor. The ridge regression is done following exact CP.
    #
    # This gives confidence intervals directly, not quantiles, so we compute it
    # for the eventual 1 - epsilon desired coverage.
    # --------------------------------------------------------------------------

    # [batch_size, n_query, 2]
    exact_intervals = np.empty((batch_size, n_query, 2))

    # Meta-learned regularization parameter.
    lambda_ = s_model.head.l2_regularizer.exp().item()

    for i in range(batch_size):
        for j in range(n_query):
            X = torch.cat([support[i], query[i, j].unsqueeze(0)], dim=0)
            Y_ = support_targets[i]
            pred = exact_cp.conf_pred(
                X=X.cpu().numpy(),
                Y_=Y_.cpu().numpy(),
                lambda_=lambda_,
                alpha=args.epsilon)
            exact_intervals[i, j] = pred

    # --------------------------------------------------------------------------
    # Step 5.
    #
    # Another baseline. This time the jackknife+ method from Barber et. al.
    # We compute and store p-values for each query.
    #
    # Note that jackknife+ doesn't give exact guarantees for 1 - epsilon (though
    # it is often conservative). We also compute the "corrected" intervals.
    # --------------------------------------------------------------------------

    # [batch_size, n_support, n_query, dim]
    jk_query = query.unsqueeze(1).repeat(1, n_support, 1, 1)
    jk_query_targets = query_targets.unsqueeze(1).repeat(1, n_support, 1)

    # [batch_size * n_support, n_query]
    jk_pred = s_model.head(
        jk_query.view(batch_size * n_support, n_query, dim),
        loo_support.view(batch_size * n_support, n_support - 1, dim),
        loo_support_targets.view(batch_size * n_support, n_support - 1))

    # [batch_size, n_support, n_query]
    jk_pred = jk_pred.view(batch_size, n_support, n_query)
    jk_loo_scores = (jk_query_targets - jk_pred).pow(2)

    # Count number of loo support scores that are more conforming (less than)
    # the corresponding loo query scores.
    jk_loo_calibration = loo_scores.unsqueeze(-1).expand(batch_size, n_support, n_query)

    # [batch_size, n_query]
    jk_pvalues = jk_loo_calibration.le(jk_loo_scores).sum(dim=1) / (n_support + 1)

    return dict(pred_quantiles=pred_quantiles.tolist(),
                query_scores=query_scores.tolist(),
                query_targets=query_targets.tolist(),
                exact_intervals=exact_intervals.tolist(),
                jk_pvalues=jk_pvalues.tolist())


def main(args):
    # --------------------------------------------------------------------------
    # A note on data sampling:
    #
    # A single "trial" will consist of each of the T + 1 tasks, with k + n
    # samples per task. Which are the "calibration" tasks and which is the "test"
    # task will be marginalized. Performance over each choice of calibration/test
    # tasks will also be marginalized. To do that, we need multiple samples per
    # task of "support" and "query". Ideally should be balanced, but we can just
    # compile a large number of samples, and it should be sufficient to resample
    # from in the conformal prediction stage.
    # --------------------------------------------------------------------------

    # Load models.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    s_model = conformal_mpn.ConformalMPN.load_from_checkpoint(args.s_ckpt).to(device)
    s_model.eval()
    q_model = quantile_snn.QuantileSNN.load_from_checkpoint(args.q_ckpt).to(device)
    q_model.eval()

    # Load data
    n_support = s_model.hparams.num_support
    dataset, indices, task_dict = s_model.load_dataset(
        dataset_file=args.dataset_file,
        features_file=args.features_file)
    sampler = conformal_mpn.FewShotSampler(
        indices=indices,
        tasks_per_batch=args.batch_size,
        num_support=n_support,
        num_query=args.num_query,
        iters=args.num_samples // args.batch_size)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=args.num_data_workers,
        collate_fn=conformal_mpn.construct_molecule_batch)

    # Mapping of task index back to string.
    idx2task = {v: k for k, v in task_dict.items()}

    # Evaluate batches and write examples to disk.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        with torch.no_grad():
            for inputs, (smiles, tasks) in tqdm.tqdm(loader, desc="evaluating"):
                outputs = predict(s_model, q_model, inputs, args)
                for i in range(args.batch_size):
                    example = dict(
                        pred_quantile=outputs["pred_quantiles"][i],
                        query_scores=outputs["query_scores"][i],
                        query_targets=outputs["query_targets"][i],
                        exact_intervals=outputs["exact_intervals"][i],
                        jk_pvalues=outputs["jk_pvalues"][i],
                        task=idx2task[tasks[i]])
                    f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_ckpt", type=str, default=None)
    parser.add_argument("--s_ckpt", type=str, default=None)
    parser.add_argument("--dataset_file", type=str, default=None)
    parser.add_argument("--features_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_query", type=int, default=250)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--num_data_workers", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="../ckpts/chembl/k=10/conformal_mpn")
    args = parser.parse_args()
    main(args)
