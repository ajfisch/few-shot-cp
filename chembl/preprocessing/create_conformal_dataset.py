"""Prepare the input files for conformal prediction."""

import argparse
import json
import os
import torch
import tqdm

from modeling import nonconformity
from modeling import quantile


def predict(s_model, q_model, inputs, args):
    """Predict quantiles and compute baselines for an input batch.

    Note that we only compute the baselines for n_query, where we assume
    n_query < n_calibration (we only need the large n for our method).

    Prediction follows steps:
       1) Compute leave-out-one scores on the support set.
       2) Predict the leave-out-none quantile.
       3) Compute the leave-out-none query scores.
       4) Change query to only have n_query values instead of n_calibration.
       5) Provide exact conformal prediction inputs.

    Args:
        s_model: NonconformityNN model.
        q_model: QuantileNN model.
        inputs: Input batch of support and query examples.

    Returns:
        pred_quantiles: [batch_size]
        query_preds: [batch_size, n_calibration]
        query_targets: [batch_size, n_calibration]
        support_encs: [batch_size, n_support, dim]
        query_encs: [batch_size, n_query, dim]
    """
    batch_size = args.batch_size
    n_calibration = args.num_calibration
    n_query = args.num_query
    n_support = s_model.hparams.num_support
    dim = s_model.hparams.enc_hidden_size

    inputs = s_model.transfer_batch_to_device(inputs, s_model.device)

    # Encode molecules.
    (query, support, support_targets), query_targets = s_model.encode(
        inputs, [batch_size, n_support, n_calibration])

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
    loo_scores = (loo_query_targets - loo_pred).abs()

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

    # [batch_size, n_calibration]
    query_preds = s_model.head(query, support, support_targets)

    # [batch_size, n_calibration]
    query_scores = (query_preds - query_targets).abs()

    # --------------------------------------------------------------------------
    # Step 4.
    #
    # Truncate queries to just those necessary for baselines.
    # --------------------------------------------------------------------------

    # [batch_size, n_query, dim]
    query = query[:, :n_query]

    return dict(pred_quantiles=pred_quantiles.tolist(),
                query_preds=query_preds.tolist(),
                query_targets=query_targets.tolist(),
                query_scores=query_scores.tolist(),
                support_encs=support.tolist(),
                support_targets=support_targets.tolist(),
                query_encs=query.tolist())


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

    args.epsilon = 1 - args.tolerance

    # Load models.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    s_model = nonconformity.NonconformityNN.load_from_checkpoint(args.s_ckpt).to(device)
    s_model.eval()
    q_model = quantile.QuantileNN.load_from_checkpoint(args.q_ckpt).to(device)
    q_model.eval()

    # Check num_calibration > num_query.
    assert(args.num_calibration > args.num_query)

    # Load data
    n_support = s_model.hparams.num_support
    dataset, indices, _ = s_model.load_dataset(
        dataset_file=args.dataset_file,
        features_file=args.features_file)
    sampler = nonconformity.FewShotSampler(
        indices=indices,
        tasks_per_batch=args.batch_size,
        num_support=n_support,
        num_query=args.num_calibration,
        iters=args.num_samples // args.batch_size)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=args.num_data_workers,
        collate_fn=nonconformity.construct_molecule_batch)

    # Meta-learned regularization parameter.
    l2_lambda = s_model.head.l2_regularizer.exp().item()

    # Evaluate batches and write examples to disk.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        with torch.no_grad():
            for inputs, (smiles, tasks) in tqdm.tqdm(loader, desc="evaluating"):
                outputs = predict(s_model, q_model, inputs, args)
                for i in range(args.batch_size):
                    example = dict(
                        pred_quantile=outputs["pred_quantiles"][i],
                        query_preds=outputs["query_preds"][i],
                        query_scores=outputs["query_scores"][i],
                        query_targets=outputs["query_targets"][i],
                        support_encs=outputs["support_encs"][i],
                        support_targets=outputs["support_targets"][i],
                        query_encs=outputs["query_encs"][i],
                        l2_lambda=l2_lambda,
                        task=tasks[i])
                    f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_ckpt", type=str, default=None)
    parser.add_argument("--s_ckpt", type=str, default="../ckpts/chembl/k=10/nonconformity/full/_ckpt_epoch_11.ckpt")
    parser.add_argument("--dataset_file", type=str, default="../data/chembl/test_molecules.csv")
    parser.add_argument("--features_file", type=str, default="../data/chembl/features/test_molecules.npy")
    parser.add_argument("--tolerance", type=float, default=0.90)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_query", type=int, default=10)
    parser.add_argument("--num_calibration", type=int, default=250)
    parser.add_argument("--num_samples", type=int, default=25000)
    parser.add_argument("--num_data_workers", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="../ckpts/chembl/k=10/conformal/q=0.90/val.jsonl")
    args = parser.parse_args()
    main(args)
