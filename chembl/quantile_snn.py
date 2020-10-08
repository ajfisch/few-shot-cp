"""Quantile prediction using a deep set NN."""

import argparse
import chemprop
import collections
import encoders
import json
import numpy as np
import os
import pytorch_lightning as pl
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

MoleculeSet = collections.namedtuple(
    "MoleculeSet", ["scores", "mols", "target", "task"])


def construct_molecule_batch(batch):
    tasks = [x.task for x in batch]

    if batch[0].mols:
        inputs = chemprop.data.data.construct_molecule_batch(
            [mol for x in batch for mol in x.mols])

        mol_graph = inputs.batch_graph()
        if mol_graph:
            mol_graph = mol_graph.get_components()

        mol_features = inputs.features()
        if mol_features:
            mol_features = torch.from_numpy(np.stack(mol_features)).float()
    else:
        # Dummy
        mol_graph = mol_features = torch.Tensor(0)

    scores = torch.Tensor([x.scores for x in batch]).float()
    targets = torch.Tensor([x.target for x in batch]).float()

    inputs = [mol_graph, mol_features, scores, targets]
    return inputs, tasks


class QuantileSNN(pl.LightningModule):
    """Few-shot DeepSet to compute quantiles."""

    def __init__(self, hparams):
        super(QuantileSNN, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        if hparams.use_adaptive_encoding:
            self.encoder = encoders.MoleculeEncoder(hparams)
        self.head = encoders.DeepSet(hparams)

    def forward(self, inputs):
        mol_graph, mol_features, scores, targets = inputs

        # [batch_size, n_support, 1]
        input_encs = [scores.unsqueeze(-1)]

        # Optionally add input encodings.
        if self.hparams.use_adaptive_encoding:
            # [batch_size * n_support, dim]
            mol_encs = self.encoder(mol_graph, mol_features)

            # [batch_size, n_support, dim]
            mol_encs = mol_encs.view(-1, self.hparams.num_support, self.hparams.enc_hidden_size)
            mol_encs = F.dropout(mol_encs, p=self.hparams.dropout, training=self.training)
            input_encs.append(mol_encs)

        input_encs = torch.cat(input_encs, dim=-1)

        # [batch_size]
        y_pred = self.head(input_encs)

        # MSE.
        loss = (targets - y_pred).pow(2)

        return dict(loss=loss, preds=y_pred)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        outputs = self.forward(inputs)
        loss = outputs["loss"].mean()
        tensorboard_logs = {"train_loss": loss.detach()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, _ = batch
        return self.forward(inputs)

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"].detach() for x in outputs], dim=0).mean()
        tensorboard_logs = {"val_loss": loss}
        tqdm_dict = tensorboard_logs
        return {"val_loss": loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, min_lr=1e-6)
        return [optimizer], [scheduler]

    @staticmethod
    def load_dataset(dataset_file, alpha, features_file=None, features_generator=None, adaptive=True):
        if features_file is not None:
            smiles_to_idx, idx_to_features = np.load(features_file, allow_pickle=True)

        dataset = []
        num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
        with open(dataset_file, "r") as f:
            for line in tqdm.tqdm(f, total=num_lines):
                line = json.loads(line)
                mols = []
                if adaptive:
                    for smiles in line["s_smiles"]:
                        mols.append(chemprop.data.MoleculeDatapoint(
                            smiles=smiles,
                            features=idx_to_features[smiles_to_idx[smiles]].copy() if features_file else None,
                            features_generator=features_generator))
                scores = line["s_scores"]
                target = np.quantile(line["f_scores"] + [np.inf], alpha, interpolation="higher")
                task = line["task"]
                dataset.append(MoleculeSet(scores, mols, target, task, alpha))

        return dataset

    def prepare_data(self):
        for split in ["train", "val"]:
            dataset = self.load_dataset(
                dataset_file=getattr(self.hparams, "%s_data" % split),
                alpha=self.hparams.alpha,
                features_file=getattr(self.hparams, "%s_features" % split),
                features_generator=self.hparams.features_generator,
                adaptive=self.hparams.use_adaptive_encoding)
            setattr(self, "%s_dataset" % split, dataset)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_molecule_batch)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_molecule_batch)
        return loader

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.register("type", "bool", pl.utilities.parsing.str_to_bool)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--gpus", type=int, nargs="+", default=None)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--checkpoint_dir", type=str, default="../ckpts/chembl/debug/quantile_snn_noadapt")
        parser.add_argument("--overwrite", type="bool", default=True)

        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_support", type=int, default=10)
        parser.add_argument("--max_epochs", type=int, default=15)

        parser.add_argument("--num_data_workers", type=int, default=20)
        parser.add_argument("--train_data", type=str, default="../ckpts/chembl/k=10/conformal_mpn/train_quantile.jsonl")
        parser.add_argument("--train_features", type=str, default="../data/chembl/features/train_molecules.npy")
        parser.add_argument("--val_data", type=str, default="../ckpts/chembl/k=10/conformal_mpn/val_quantile.jsonl")
        parser.add_argument("--val_features", type=str, default="../data/chembl/features/val_molecules.npy")

        parser.add_argument("--alpha", type=float, default=0.9)
        parser.add_argument("--use_adaptive_encoding", type="bool", default=True)
        parser.add_argument("--features_generator", type=str, nargs="+", default=None)
        parser.add_argument("--use_mpn_features", type="bool", default=True)
        parser.add_argument("--use_mol_features", type="bool", default=True)
        parser.add_argument("--mpn_hidden_size", type=int, default=256)
        parser.add_argument("--mol_features_size", type=int, default=200)
        parser.add_argument("--ffnn_hidden_size", type=int, default=256)
        parser.add_argument("--num_ffnn_layers", type=int, default=3)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--mpn_depth", type=int, default=2)
        parser.add_argument("--undirected_mpn", type="bool", default=False)
        parser.add_argument("--enc_hidden_size", type=int, default=128)
        parser.add_argument("--set_hidden_size", type=int, default=256)

        return parser


def main(args):
    pl.seed_everything(args.seed)
    model = QuantileSNN(hparams=args)
    print(model)
    if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir):
        if not args.overwrite:
            raise RuntimeError("Experiment directory is not empty.")
        else:
            shutil.rmtree(args.checkpoint_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, "weights.pt"),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min")
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=os.path.join(args.checkpoint_dir, "logs"))
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        terminate_on_nan=True)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = QuantileSNN.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
