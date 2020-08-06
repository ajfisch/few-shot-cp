"""Create quantile data."""

import argparse
import glob
import json
import models
import os
import regex
import torch
import tqdm


def get_ckpt(fold_dir, fold_ckpts):
    all_ckpts = []
    for fold_ckpt in fold_ckpts:
        all_ckpts.extend(glob.glob(fold_ckpt))
    fold_idx = int(regex.match(".+?fold_(\d+).*", fold_dir).groups()[0])
    fold_ckpt = None
    for ckpt in all_ckpts:
        if "fold_%d" % fold_idx in ckpt:
            fold_ckpt = ckpt
            break
    assert(fold_ckpt)
    return fold_idx, fold_ckpt


def get_loader(fold_idx, fold_dir, num_support, num_query, num_workers, num_samples):
    if fold_dir[-1] == "/":
        fold_dir = fold_dir[:-1]
    fold_dir = os.path.dirname(fold_dir)
    data_files = os.path.join(fold_dir, "fold_[!%d]" % fold_idx, "*.csv")
    features_files = os.path.join(fold_dir, "fold_[!%d]" % fold_idx, "*.npy")
    datasets = models.ProtoNet.load_datasets(
        file_pattern=data_files,
        features_pattern=features_files)
    loader = models.ProtoNet.few_shot_dataloader(
        datasets=datasets,
        num_query=num_query,
        num_support=num_support,
        num_workers=num_workers,
        max_examples=num_samples)
    return loader


def run_fold(fold_dir, args):
    fold_idx, ckpt = get_ckpt(fold_dir, args.fold_ckpts)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.ProtoNet.load_from_checkpoint(ckpt).to(device)
    model.eval()
    model.hparams.num_support = args.num_support
    model.hparams.num_query = args.num_query
    dataloader = get_loader(
        fold_idx=fold_idx,
        fold_dir=fold_dir,
        num_support=args.num_support,
        num_query=args.num_query,
        num_workers=args.num_workers,
        num_samples=args.num_samples)
    with open(os.path.join(args.output_dir, "fold_%d.json" % fold_idx), "w") as f:
        for batch in tqdm.tqdm(dataloader, desc="running inference"):
            batch, (smiles, task) = batch
            batch = model.transfer_batch_to_device(batch, device)

            output = model.get_few_shot(batch)
            output_scores = output["loss"].cpu().detach().numpy().tolist()
            output_labels = output["targets"].cpu().detach().numpy().tolist()

            inputs = model.get_few_shot_calibration(batch)
            input_scores = inputs["loss"].cpu().detach().numpy().tolist()
            input_labels = inputs["targets"].cpu().detach().numpy().tolist()

            data_point = dict(
                task=task,
                smiles=smiles[:model.hparams.num_support * model.hparams.num_classes],
                input_scores=input_scores,
                input_labels=input_labels,
                output_scores=output_scores,
                output_labels=output_labels)

            f.write(json.dumps(data_point) + "\n")


def main(args):
    fold_dirs = glob.glob(args.fold_dirs)
    os.makedirs(args.output_dir, exist_ok=True)
    for fold in tqdm.tqdm(fold_dirs, desc="running folds"):
        run_fold(fold, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_ckpts", type=str, nargs="+",
                        default=["../ckpts/chembl/few_shot/random/fold_[0-9]*/"
                                 "lightning_logs/version_0/checkpoints/*.ckpt"])
    parser.add_argument("--fold_dirs", type=str,
                        default="../data/chembl/random/train_folds/fold_[0-9]*/")
    parser.add_argument("--output_dir", type=str,
                        default="../data/chembl/random/quantiles/proto/train_wtask/k=10/")
    parser.add_argument("--num_support", type=int, default=10)
    parser.add_argument("--num_query", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()
    main(args)
