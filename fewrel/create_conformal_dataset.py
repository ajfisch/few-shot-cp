"""Prepare the input files for conformal prediction."""

import pprint
import argparse
import json
import numpy as np
import os
import torch
import tqdm

import quantile_snn
from create_quantile_dataset import euclidean_dist, load_model

from FewRel.fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
from FewRel.fewshot_re_kit.data_loader import get_loader_wclass
from FewRel.models.proto import Proto
from FewRel.models.gnn import GNN
from FewRel.models.snail import SNAIL
from FewRel.models.metanet import MetaNet
from FewRel.models.siamese import Siamese
from FewRel.models.pair import Pair
from FewRel.models.d import Discriminator
from FewRel.models.mtb import Mtb



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
    support, query, labels = inputs

    # [n_way, n_support, model.hidden_size]
    support_encodings = s_model.sentence_encoder(support).view(args.n_way, args.n_support, -1)

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
        # [n_way, model.hidden_size]
        query_emb = support_encodings[:, i:i + 1, :].squeeze()

        # Support prototypes.
        # [n_way, model.hidden_size]
        support_proto = support_encodings.index_select(
            1, support_idx).mean(dim=1)

        s_logits = euclidean_dist(query_emb, support_proto)
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

    # [n_way, model.hidden_size]
    class_prototypes = support_encodings.index_select(
        1, support_idx).mean(dim=1)

    # [n_calibration, model.hidden_size]
    query_emb = s_model.sentence_encoder(query)

    # [n_calibration * n_way, n_way]
    query_scores = euclidean_dist(query_emb, class_prototypes)
    query_probs = (-1 * torch.softmax(query_scores, dim=-1))

    # --------------------------------------------------------------------------
    # step 4.
    #
    # truncate queries to just those necessary for baselines.
    # --------------------------------------------------------------------------

    # [n_query * n_way]
    exact_labels = labels.view(args.n_way, args.n_calibration)[:,:args.n_query].reshape(-1)

    # [n_way, n_query, mode.hidden_size]
    exact_query = query_emb.view(args.n_way, args.n_calibration, -1)[:,:args.n_query]

    # --------------------------------------------------------------------------
    # Step 5.
    #
    # Exact conformal prediction baseline. In order to preserve exchangability,
    # each query is added to the support set.
    #
    # This gives the prediction set directly, not quantiles, so we compute it
    # for the eventual alpha desired coverage.
    # --------------------------------------------------------------------------

    #all_pred_sets = []
    all_pred_probs = []
    all_non_comfs = []
    all_labels = []
    for i in range(args.n_query):
        for j in range(args.n_way):
            pred_set = []
            pred_score = []
            non_comf = []
            for m in range(args.n_way):
                # Include the query itself in the prototype (moving avg).
                class_prototypes[m] = (class_prototypes[m] * args.n_support +
                                       exact_query[j, i]) / \
                                       (args.n_support + 1)

                # Extracting the non-conformity scores for the support points
                # of this class.
                # [n_support, n_way]
                sup_dists = euclidean_dist(support_encodings[m,:,:], class_prototypes)

                # [n_support]
                non_comf_scores = -1 * torch.softmax(sup_dists, dim=-1)[:,m]

                # [n_way]
                out_dists = euclidean_dist(
                    exact_query[j, i].unsqueeze(0),
                    class_prototypes).squeeze()

                # [n_way]
                out_probs = (-1 * torch.softmax(out_dists, dim=-1)).tolist()

                # Revert the prototype (removing query).
                class_prototypes[m] = (class_prototypes[m] *
                                       (args.n_support + 1) -
                                       exact_query[j, i]) / args.n_support

                pred_score.append(out_probs[m])
                non_comf.append(non_comf_scores.tolist())

            #all_pred_sets.append(pred_set)
            all_pred_probs.append(pred_score)
            all_non_comfs.append(non_comf)
            all_labels.append(j)

    # Transpose targets to be 0, 1, 2, ..., n_way-1, 0, 1, 2, ...
    query_targets = labels.view(args.n_way, args.n_calibration).transpose(0,1).reshape(-1)
    query_probs = query_probs.view(args.n_way, args.n_calibration, -1).transpose(0,1).reshape(-1, args.n_way)
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

    options = vars(args)
    printer = pprint.PrettyPrinter()
    printer.pprint(options)

    # Load models.
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    s_model, sentence_encoder = load_model(args.s_ckpt, device, args)
    s_model.eval()

    q_model = quantile_snn.QuantileSNN.load_from_checkpoint(args.q_ckpt).to(device)
    q_model.eval()

    # Check n_calibration > n_query.
    assert(args.n_calibration > args.n_query)

    # Load data.
    # n_calibration instead of n_query so we'll have enough queries from
    # the calibration tasks.
    test_loader = get_loader_wclass('val_wiki', sentence_encoder,
            N=args.n_way, K=args.n_support, Q=args.n_calibration,
            batch_size=args.batch_size, root=args.full_path, num_workers=args.workers)

    # Evaluate batches and write examples to disk.
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        with torch.no_grad():
            for it in tqdm.tqdm(range(args.n_episodes)):
                batch = next(test_loader)
                inputs, tasks = batch[:3], batch[3]
                for inp in inputs[:2]:
                    for k in inp:
                        inp[k] = inp[k].to(device)
                inputs[2] = inputs[2].to(device)

                outputs = predict(s_model, q_model, inputs, args)
                example = dict(
                    pred_quantile=outputs["pred_quantiles"],
                    query_scores=outputs["query_scores"],
                    query_targets=outputs["query_targets"],
                    #exact_intervals=outputs["exact_intervals"],
                    exact_pred_scores=outputs["exact_pred_scores"],
                    exact_non_comfs=outputs["exact_non_comfs"],
                    task="_".join([str(t) for t in sorted(tasks[::args.n_calibration])])
                    )
                f.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_ckpt", type=str, default="results/10_way/quantile_snn/q=0.80/weights.pt.ckpt")
    parser.add_argument("--s_ckpt", type=str, default="FewRel/checkpoint/proto-cnn-train_wiki-val_wiki-10-10.pth.tar")
    parser.add_argument('--full_path', type=str, default="FewRel/data/", help='path to dir with full data json files containing train/dev/test examples')
    parser.add_argument('--n_episodes', default=2000, type=int, help='Number of episodes to average')
    parser.add_argument("--n_support", type=int, default=16)
    parser.add_argument('--n_way', default=10, type=int, help='Number of classes per episode')
    parser.add_argument("--alpha", type=float, default=0.80)
    parser.add_argument("--n_query", type=int, default=5)
    parser.add_argument("--n_calibration", type=int, default=200)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--output_file", type=str, default="results/10_way/quantile_snn/0.80/conf_val.jsonl")

    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: cnn or bert or roberta')
    parser.add_argument('--batch_size', default=1, type=int,
            help='batch size')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--dot', action='store_true',
           help='use dot instead of L2 distance for proto')
    parser.add_argument('--pair', action='store_true',
           help='use pair model')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')

    args = parser.parse_args()
    main(args)
