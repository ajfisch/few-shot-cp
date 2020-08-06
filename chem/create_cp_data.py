"""Create conformal prediction data."""

import argparse
import json
import models
import os
import torch
import tqdm


def get_loader(quantile, data_files, features_files, encode_molecules, batch_size, num_workers):
    datasets = models.DeepSet.load_datasets(
        quantile=quantile,
        file_pattern=data_files,
        features_pattern=features_files)
    loader = models.DeepSet.quantile_dataloader(
        datasets=datasets,
        encode_molecules=encode_molecules,
        batch_size=batch_size,
        shuffle=False)
    return loader


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = models.DeepSet.load_from_checkpoint(args.ckpt).to(device)
    model.eval()
    dataloader = get_loader(
        quantile=model.hparams.quantile,
        data_files=args.input_files,
        features_files=args.input_features,
        encode_molecules=model.hparams.encode_molecules,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "data.json"), "w") as f:
        for batch in tqdm.tqdm(dataloader, desc="running inference"):
            batch, metadata = batch
            batch = model.transfer_batch_to_device(batch, device)
            output = model.get_few_shot(batch)
            quantiles = output["preds"]
            for i in range(len(quantiles)):
                data_point = json.loads(metadata[i])
                data_point["quantile"] = quantiles[i].item()
                f.write(json.dumps(data_point) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default=("../ckpts/chembl/quantiles/random/q=0.90,l=mse,mol=false/"
                                 "lightning_logs/version_0/checkpoints/epoch=18.ckpt"))
    parser.add_argument("--input_files", type=str,
                        default="../data/chembl/random/quantiles/proto/val_wtask_cv/k=10/*.json")
    parser.add_argument("--input_features", type=str,
                        default="../data/chembl/random/val/*.npy")
    parser.add_argument("--output_dir", type=str,
                        default="../data/chembl/random/conformal/proto/val/k=10/q=0.90,l=mse,mol=false")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
