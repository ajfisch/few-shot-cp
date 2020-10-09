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
    dataset, indices, tasks = model.load_dataset(
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
            inputs = model.transfer_batch_to_device(inputs, model.device)

            # Reshape smiles and tasks to [batch_size, n_support + n_query]
            smiles = np.array(smiles).reshape(args.batch_size, -1).tolist()
            tasks = np.array(tasks).reshape(args.batch_size, -1).tolist()

            # Encode molecules.
            (query, support, support_targets), query_targets = model.encode(
                inputs, [args.batch_size, model.hparams.num_support, args.num_query])

            # === Evaluate on Kth support using K-1 support set. ===
            # Construct K-fold batch.
            s_batch = [[] for _ in range(4)]
            for i in range(model.hparams.num_support):
                support_idx = [j for j in range(model.hparams.num_support) if j != i]
                support_idx = torch.LongTensor(support_idx).to(model.device)
                s_batch[0].append(support[:, i:i + 1, :].view(
                    -1, 1, model.hparams.enc_hidden_size))
                s_batch[1].append(support.index_select(1, support_idx).view(
                    -1, model.hparams.num_support - 1, model.hparams.enc_hidden_size))
                s_batch[2].append(support_targets.index_select(1, support_idx).view(
                    -1, model.hparams.num_support - 1))
                s_batch[3].append(support_targets[:, i:i + 1].view(-1, 1))
            s_batch = [torch.cat(s, dim=0) for s in s_batch]

            # [batch_size * n_support, 1]
            s_pred = model.head(*s_batch[:3])

            # [batch_size, n_support]
            s_scores = (s_batch[3] - s_pred).pow(2)
            s_scores = s_scores.view(-1, model.hparams.num_support).tolist()

            # === Evaluate on all queries using full support set. ===
            # [batch_size, n_query]
            f_pred = model.head(query, support, support_targets)

            # [n_query]
            f_scores = (query_targets - f_pred).pow(2).squeeze(0).tolist()

            for i in range(args.batch_size):
                all_outputs.append(
                    dict(s_scores=s_scores[i],
                         f_scores=f_scores[i],
                         task=tasks[i][0],
                         s_smiles=smiles[i][:model.hparams.num_support]))

    return all_outputs


def main(args):
    # all_outputs = []
    # for ckpt in tqdm.tqdm(args.fold_ckpts, desc="evaluating folds"):
    #     fold_outputs = run_fold(
    #         dataset_file=os.path.join(args.data_dir, "val_f%d_molecules.csv" % i),
    #         features_file=os.path.join(args.features_dir, "val_f%d_molecules.npy" % i),
    #         ckpt=ckpt,
    #         args=args)
    #     all_outputs.extend(fold_outputs)

    # os.makedirs(args.output_dir, exist_ok=True)
    # with open(os.path.join(args.output_dir, "train_quantile.jsonl"), "w") as f:
    #     for output in all_outputs:
    #         f.write(json.dumps(output, sort_keys=True) + "\n")

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
