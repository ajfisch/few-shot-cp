"""Create quantile prediction training data."""

import argparse
import conformal_mpn
import json
import numpy as np
import os
import torch
import tqdm


def run_fold(dataset_file, features_file, ckpt, args):
    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = conformal_mpn.ConformalMPN.load_from_checkpoint(ckpt).to(device)
    model.eval()

    # Load data
    dataset, indices, _ = model.load_dataset(
        dataset_file=dataset_file,
        features_file=features_file)
    sampler = conformal_mpn.FewShotSampler(
        indices=indices,
        tasks_per_batch=args.batch_size,
        num_support=model.hparams.num_support,
        num_query=args.num_query,
        iters=args.num_samples_per_fold // args.batch_size)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=args.num_data_workers,
        collate_fn=conformal_mpn.construct_molecule_batch)

    # Evaluate
    all_outputs = []
    with torch.no_grad():
        for inputs, (smiles, tasks) in tqdm.tqdm(loader, desc="evaluating fold"):
            batch_size = args.batch_size
            n_support = model.hparams.num_support
            dim = model.hparams.enc_hidden_size

            inputs = model.transfer_batch_to_device(inputs, model.device)

            # Reshape smiles and tasks to [batch_size, n_support + n_query]
            smiles = np.array(smiles).reshape(args.batch_size, -1).tolist()
            tasks = np.array(tasks).reshape(args.batch_size, -1).tolist()

            # Encode molecules.
            (query, support, support_targets), query_targets = model.encode(
                inputs, [args.batch_size, model.hparams.num_support, args.num_query])

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
                support_idx = torch.LongTensor(support_idx).to(model.device)

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
            loo_pred = model.head(
                loo_queries.view(batch_size * n_support, 1, dim),
                loo_support.view(batch_size * n_support, n_support - 1, dim),
                loo_support_targets.view(batch_size * n_support, n_support - 1))

            # [batch_size, n_support, 1]
            loo_pred = loo_pred.view(batch_size, n_support, 1)
            loo_scores = (loo_query_targets - loo_pred).pow(2)

            # [batch_size, n_support]
            loo_scores = loo_scores.view(batch_size, n_support).tolist()

            # --------------------------------------------------------------------------
            # Step 2.
            #
            # Using the nonconformity model with the *full* support set (all K examples),
            # predict nonconformity scores on the query points.
            # --------------------------------------------------------------------------
            query_pred = model.head(query, support, support_targets)

            # [n_query]
            query_scores = (query_targets - query_pred).pow(2).squeeze(0).tolist()

            for i in range(args.batch_size):
                all_outputs.append(
                    dict(s_scores=loo_scores[i],
                         f_scores=query_scores[i],
                         task=tasks[i][0],
                         s_smiles=smiles[i][:n_support]))

    return all_outputs


def main(args):
    all_outputs = []
    for i, ckpt in enumerate(tqdm.tqdm(args.fold_ckpts, desc="evaluating folds")):
        fold_outputs = run_fold(
            dataset_file=os.path.join(args.data_dir, "val_f%d_molecules.csv" % i),
            features_file=os.path.join(args.features_dir, "val_f%d_molecules.npy" % i),
            ckpt=ckpt,
            args=args)
        all_outputs.extend(fold_outputs)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "train_quantile.jsonl"), "w") as f:
        for output in all_outputs:
            f.write(json.dumps(output, sort_keys=True) + "\n")

    val_outputs = run_fold(
        dataset_file=os.path.join(args.data_dir, "val_molecules.csv"),
        features_file=os.path.join(args.features_dir, "val_molecules.npy"),
        ckpt=args.full_ckpt,
        args=args)

    with open(os.path.join(args.output_dir, "val_quantile.jsonl"), "w") as f:
        for output in val_outputs:
            f.write(json.dumps(output, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_ckpts", type=str, nargs="+",
                        default=["../ckpts/chembl/k=10/conformal_mpn/0/_ckpt_epoch_12.ckpt",
                                 "../ckpts/chembl/k=10/conformal_mpn/1/_ckpt_epoch_10.ckpt",
                                 "../ckpts/chembl/k=10/conformal_mpn/2/_ckpt_epoch_13.ckpt",
                                 "../ckpts/chembl/k=10/conformal_mpn/3/_ckpt_epoch_1.ckpt",
                                 "../ckpts/chembl/k=10/conformal_mpn/4/_ckpt_epoch_1.ckpt"])
    parser.add_argument("--full_ckpt", type=str,
                        default="../ckpts/chembl/k=10/conformal_mpn/full/_ckpt_epoch_11.ckpt")
    parser.add_argument("--data_dir", type=str, default="../data/chembl")
    parser.add_argument("--features_dir", type=str, default="../data/chembl/features")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_query", type=int, default=250)
    parser.add_argument("--num_samples_per_fold", type=int, default=20000)
    parser.add_argument("--num_data_workers", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="../ckpts/chembl/k=10/conformal_mpn")
    args = parser.parse_args()
    main(args)
