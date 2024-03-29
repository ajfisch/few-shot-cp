"""Prepare the input files for conformal prediction."""

import argparse
import json
import numpy as np
import os
import torch
import tqdm

import quantile_snn

from prototypical_networks.samplers.episodic_batch_sampler import EpisodicBatchSamplerWithClass
from prototypical_networks.dataloaders.mini_imagenet_loader import MiniImageNet
from prototypical_networks.models.convnet_mini import ConvNet
from prototypical_networks.utils import euclidean_dist
from torch.utils.data import DataLoader


def predict(s_model, q_model, inputs, args):
    """Predict quantiles and compute baselines for an input batch.

    Note that we only compute the baselines for n_query, where we assume
    n_query < n_calibration (we only need the large n for our method).

    Prediction follows steps:
       1) Compute leave-out-one scores on the support set.
       2) Predict the leave-out-one quantile.
       3) Compute the leave-out-one query scores.
       4) Change query to only have n_query values instead of n_calibration.
       5) Compute the exact conformal prediction baseline.

    Args:
        s_model: ConformalMPN model.
        q_model: QuantileSNN model.
        inputs: Input batch of support and query examples.

    Returns:
        pred_quantiles: [n_way]
        query_scores: [n_way * n_calibration, n_way]
        query_targets: [n_way * n_calibration]
        exact_pred_scores: [n_way * n_query, n_way]
        exact_non_comfs: [n_way * n_query, n_way, n_support]
    """
    p = args.n_support * args.n_way
    # data_support: [n_support * n_way, 3, 84, 84]
    # data_query: [n_calibration * n_way, 3, 84, 84]
    data_support, data_query = inputs[:p], inputs[p:]

    # Encode images.
    # [n_support, n_way, s_model.out_channels]
    support_encodings = s_model(data_support).view(args.n_support, args.n_way, -1)

    # --------------------------------------------------------------------------
    # Step 1.
    #
    # Compute leave-out-one scores on the support set.
    # These "scores" will be used as inputs to the quantile predictor.
    # --------------------------------------------------------------------------

    # [n_way]
    all_s_scores = [[] for _ in range(args.n_way)]
    all_s_probs = [[] for _ in range(args.n_way)]
    for i in range(args.n_support):
        # All indices *except* this one for support.
        support_idx = [j for j in range(args.n_support) if j != i]
        support_idx = torch.LongTensor(support_idx).to(
            support_encodings.device)

        # Query.
        # Held out i-th support example is the "query".
        # [n_way, model.output_channels]
        query = support_encodings[i:i + 1, :, :].squeeze()

        # Support prototypes.
        # [n_way, model.output_channels]
        support_proto = support_encodings.index_select(
            0, support_idx).mean(dim=0)
        s_logits = euclidean_dist(query, support_proto)
        s_scores = s_logits.diag().tolist()
        s_probs = (-1 * torch.softmax(s_logits, dim=1).diag()).tolist()
        for i, s in enumerate(s_scores):
            all_s_scores[i].append(s)
            all_s_probs[i].append(s_probs[i])

    # [n_way, n_support]
    #loo_scores = torch.tensor(all_s_scores).to(q_model.device)
    loo_probs = torch.tensor(all_s_probs).to(q_model.device)


    # --------------------------------------------------------------------------
    # Step 2.
    #
    # Using the leave-out-one scores on the support set, predict the alpha
    # quantile using the given quantile predictor model.
    #
    # We just save this prediction for now---we'll conformalize it later.
    # --------------------------------------------------------------------------

    # [n_way]
    pred_quantiles = q_model.predict(scores=loo_probs)

    # --------------------------------------------------------------------------
    # Step 3.
    #
    # Using the nonconformity model with the *full* support set (all K examples),
    # predict nonconformity scores on the query points.
    #
    # Again, we just save this prediction for now. After the quantile above is
    # conformalized, we'll compare these scores to make predictions.
    # --------------------------------------------------------------------------

    # [n_way, hidden_size]
    class_prototypes = s_model(data_support).view(
        args.n_support, args.n_way, -1).mean(dim=0)

    # [n_way * n_calibration, n_way]
    query_scores = euclidean_dist(s_model(data_query), class_prototypes)
    query_targets = torch.arange(args.n_way).long().repeat(args.n_calibration)

    # [n_way * n_calibration, n_way]
    query_probs = (-1 * torch.softmax(query_scores, dim=-1))

    # --------------------------------------------------------------------------
    # step 4.
    #
    # truncate queries to just those necessary for baselines.
    # --------------------------------------------------------------------------

    # [n_query, 3, 84, 84]
    data_query = data_query[:args.n_query * args.n_way]

    # --------------------------------------------------------------------------
    # Step 5.
    #
    # Exact conformal prediction baseline. In order to preserve exchangability,
    # each query is added to the support set.
    #
    # This gives the prediction set directly, not quantiles, so we compute it
    # for the eventual alpha desired coverage.
    # --------------------------------------------------------------------------

    query_embeds = s_model(data_query).view(args.n_query, args.n_way, -1)

    # [n_support, n_way, hidden_dim]
    support_embeds = s_model(data_support).view(
        args.n_support, args.n_way, -1)

    sup_targets = torch.arange(args.n_way).long().repeat(args.n_support)

    # [n_way, hidden_dim]
    class_prototypes = support_embeds.mean(dim=0)

    #all_pred_sets = []
    all_pred_probs = []
    all_non_comfs = []
    for i in range(args.n_query):
        for j in range(args.n_way):
            pred_set = []
            pred_score = []
            non_comf = []
            for m in range(args.n_way):
                # Include the query itself in the prototype (moving avg).
                class_prototypes[m] = (class_prototypes[m] * args.n_support +
                                       query_embeds[i, j]) / \
                                       (args.n_support + 1)

                # Extracting the non-conformity scores for the support points
                # of this class.
                # [n_support, n_way]
                sup_dists = euclidean_dist(support_embeds[:,m,:], class_prototypes)

                # [n_support]
                non_comf_scores = -1 * torch.softmax(sup_dists, dim=-1)[:,m]

                # [n_way]
                out_dists = euclidean_dist(
                    query_embeds[i, j].unsqueeze(0),
                    class_prototypes).squeeze()

                # [n_way]
                out_probs = (-1 * torch.softmax(out_dists, dim=-1)).tolist()

                # Legacy (constract prediction set).
                if False:
                    # Include k + 1 point (inf) and compute the quantile.
                    quantile = np.quantile(
                        non_comf_scores.tolist() + [np.inf],
                        args.alpha,
                        interpolation="higher")

                    if out_probs[m] <= quantile:
                        pred_set.append(m)
                        pred_score.append(out_probs[m].tolist())

                # Revert the prototype (removing query).
                class_prototypes[m] = (class_prototypes[m] *
                                       (args.n_support + 1) -
                                       query_embeds[i, j]) / args.n_support

                pred_score.append(out_probs[m])
                non_comf.append(non_comf_scores.tolist())

            #all_pred_sets.append(pred_set)
            all_pred_probs.append(pred_score)
            all_non_comfs.append(non_comf)

    return dict(pred_quantiles=pred_quantiles.tolist(),
                query_scores=query_probs.tolist(),
                query_targets=query_targets.tolist(),
                #exact_intervals=all_pred_sets,
                exact_pred_scores=all_pred_probs,
                exact_non_comfs=all_non_comfs,
                )


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

    checkpoint = torch.load(args.s_ckpt)
    s_model = ConvNet()
    s_model.load_state_dict(checkpoint['state_dict'])
    s_model = s_model.to(device)
    s_model.eval()

    q_model = quantile_snn.QuantileSNN.load_from_checkpoint(args.q_ckpt).to(device)
    q_model.eval()

    # Check n_calibration > n_query.
    assert(args.n_calibration > args.n_query)

    # Load data
    test_dataset = MiniImageNet(args.split, args.full_path, args.images_path)

    # n_calibration instead of n_query so we'll have enough queries from
    # the calibration tasks.
    test_sampler = EpisodicBatchSamplerWithClass(test_dataset.labels,
                                                 args.n_episodes, args.n_way,
                                                 args.n_support + args.n_calibration)

    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler,
                             num_workers=args.num_data_workers, pin_memory=True)

    # Evaluate batches and write examples to disk.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        with torch.no_grad():
            for n_episode, batch in tqdm.tqdm(enumerate(test_loader, 1), total=args.n_episodes):
                inputs, tasks = [_.cuda(non_blocking=True) for _ in batch]
                outputs = predict(s_model, q_model, inputs, args)
                example = dict(
                    pred_quantile=outputs["pred_quantiles"],
                    query_scores=outputs["query_scores"],
                    query_targets=outputs["query_targets"],
                    #exact_intervals=outputs["exact_intervals"],
                    exact_pred_scores=outputs["exact_pred_scores"],
                    exact_non_comfs=outputs["exact_non_comfs"],
                    task="_".join([str(t) for t in sorted(tasks[:args.n_way].tolist())]))
                f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_ckpt", type=str, default="results/10_way/quantile_snn/q=0.80/weights.pt.ckpt")
    parser.add_argument("--s_ckpt", type=str, default="models_trained/mini_imagenet_10_shot_10_way/model_best_acc.pth.tar")

    parser.add_argument('--images_path', type=str, default="mini_imagenet", help='path to parent dir with "images" dir.')
    parser.add_argument('--full_path', type=str, default="mini_imagenet", help='path to dir with full data csv files containing train/dev/test examples')
    parser.add_argument('--n_episodes', default=2000, type=int, help='Number of episodes to average')
    parser.add_argument("--n_support", type=int, default=16)
    parser.add_argument('--n_way', default=10, type=int, help='Number of classes per episode')
    parser.add_argument("--alpha", type=float, default=0.70)
    parser.add_argument("--n_query", type=int, default=5)
    parser.add_argument("--n_calibration", type=int, default=200)
    parser.add_argument("--num_data_workers", type=int, default=30)
    parser.add_argument("--output_file", type=str, default="results/10_way/quantile_snn/q=0.80/conf_val.jsonl")
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    main(args)
