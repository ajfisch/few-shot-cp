"""Create quantile data."""

import argparse
import json
import models
import os
import torch
import tqdm


def get_loader(data_files, features_files, num_support, num_query, num_workers, num_samples):
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


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.ProtoNet.load_from_checkpoint(args.ckpt).to(device)
    model.eval()
    model.hparams.num_support = args.num_support
    model.hparams.num_query = args.num_query
    dataloader = get_loader(
        data_files=args.input_files,
        features_files=args.input_features,
        num_support=args.num_support,
        num_query=args.num_query,
        num_workers=args.num_workers,
        num_samples=args.num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "data.json"), "w") as f:
        for batch in tqdm.tqdm(dataloader, desc="running inference"):
            batch, (smiles, task) = batch
            batch = model.transfer_batch_to_device(batch, device)

            output = model.get_few_shot(batch)
            output_scores = output["loss"].cpu().detach().numpy().tolist()
            output_labels = output["targets"].cpu().detach().numpy().tolist()

            inputs = model.get_few_shot_calibration(batch)
            input_scores = inputs["loss"].cpu().detach().numpy().tolist()
            input_labels = inputs["targets"].cpu().detach().numpy().tolist()
            cv_scores = inputs["cv_loss"].cpu().detach().numpy().tolist()
            cv_labels = inputs["cv_targets"].cpu().detach().numpy().tolist()

            data_point = dict(
                task=task,
                smiles=smiles[:model.hparams.num_support * model.hparams.num_classes],
                input_scores=input_scores,
                input_labels=input_labels,
                output_scores=output_scores,
                output_labels=output_labels,
                cv_scores=cv_scores,
                cv_labels=cv_labels)

            f.write(json.dumps(data_point) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=("../ckpts/chembl/few_shot/random/all_folds/"
                                 "lightning_logs/version_0/checkpoints/epoch=9.ckpt"))
    parser.add_argument("--input_files", type=str,
                        default="../data/chembl/random/val/*.csv")
    parser.add_argument("--input_features", type=str,
                        default="../data/chembl/random/val/*.npy")
    parser.add_argument("--output_dir", type=str,
                        default="../data/chembl/random/quantiles/proto/val_wtask_cv/k=10")
    parser.add_argument("--num_support", type=int, default=10)
    parser.add_argument("--num_query", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=10000)
    args = parser.parse_args()
    main(args)
