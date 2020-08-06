"""Few-Shot learning for ChEMBL."""

import argparse
import collections
import json
import functools
import glob
import multiprocessing

import molecules
import models
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
import chemprop


QuantileDatapoint = collections.namedtuple(
    "QuantileDatapoint", ["mols", "scores", "quantiles", "metadata"])


def construct_molecule_batch(batch, encode_molecules=True):
    if encode_molecules:
        mols = [x for y in batch for x in y.mols]
        mol_inputs = chemprop.data.data.construct_molecule_batch(mols)
        mol_graph = mol_inputs.batch_graph()
        if mol_graph:
            mol_graph = mol_graph.get_components()
        mol_features = mol_inputs.features()
        if mol_features:
            mol_features = torch.from_numpy(np.stack(mol_features)).float()
        mol_targets = mol_inputs.targets()
        if mol_targets:
            mol_targets = torch.Tensor(mol_targets).view(-1).float()
        mol_inputs = [mol_graph, mol_features, mol_targets]
    else:
        mol_inputs = None

    scores = torch.stack([y.scores for y in batch], dim=0)
    quantiles = torch.stack([y.quantiles for y in batch], dim=0)
    metadata = [y.metadata for y in batch]

    return (mol_inputs, scores, quantiles), metadata


class MoleculeQuantileDataset(torch.utils.data.Dataset):

    def __init__(self, molecules, scores, quantiles, metadata):
        self.molecules = molecules
        self.scores = scores
        self.quantiles = quantiles
        self.metadata = metadata

    def __getitem__(self, idx):
        return QuantileDatapoint(
            mols=self.molecules[idx],
            scores=self.scores[idx],
            quantiles=self.quantiles[idx],
            metadata=self.metadata[idx])

    def __len__(self):
        return len(self.quantiles)


class ChEMBLQuantile(pl.LightningModule):

    def __init__(self, hparams):
        super(ChEMBLQuantile, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        if hparams.encode_molecules:
            if hparams.warm_start_encoder:
                saved_model = models.ProtoNet.load_from_checkpoint(hparams.encoder_ckpt)
                self.encoder = saved_model.encoder
            else:
                self.encoder = molecules.MoleculeEncoder(hparams)

    def get_few_shot(self, inputs):
        mol_inputs, score_inputs, targets = inputs

        if self.hparams.encode_molecules:
            mol_graph, mol_features, _ = mol_inputs
            mol_encs = self.encoder(mol_graph, mol_features)

            hidden_size = self.hparams.enc_hidden_size
            classes = self.hparams.num_classes
            support = self.hparams.num_support

            # [batch_size, classes, support, hidden_size]
            support_encs = mol_encs.view(-1, classes, support, hidden_size)
        else:
            support_encs = None

        return self.forward(support_encs, score_inputs, targets)

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        outputs = self.get_few_shot(inputs)
        loss = outputs["loss"].mean()
        tensorboard_logs = {"train_loss": loss.detach()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, _ = batch
        outputs = self.get_few_shot(inputs)
        return outputs

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"] for x in outputs]).mean().item()
        tensorboard_logs = {"val_loss": loss}
        tqdm_dict = tensorboard_logs
        return {"val_loss": loss,
                "progress_bar": tqdm_dict,
                "log": tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def quantile_dataloader(datasets, batch_size=1, num_workers=0, encode_molecules=True, shuffle=True):
        dataset = torch.utils.data.ConcatDataset(datasets)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=functools.partial(construct_molecule_batch, encode_molecules=encode_molecules))
        return loader

    @staticmethod
    def _load(path, quantile, features_paths=None, features_generator=None):
        # Inefficient but temporary.
        if features_paths is not None:
            np.random.shuffle(features_paths)
            smiles_to_features = {}
            for features_path in features_paths:
                smiles_to_idx, idx_to_features = np.load(features_path, allow_pickle=True)
                for smiles, idx in smiles_to_idx.items():
                    smiles_to_features[smiles] = idx_to_features[idx]

        metadata = []
        input_molecules = []
        input_scores = []
        output_quantiles = []
        with open(path, "r") as f:
            for line in f:
                metadata.append(line)
                line = json.loads(line)
                line["input_labels"] = [x for y in line["input_labels"] for x in y]
                line["input_scores"] = [x for y in line["input_scores"] for x in y]
                mols, scores = [], []
                for idx in np.argsort(line["input_labels"]):
                    mols.append(chemprop.data.MoleculeDatapoint(
                        smiles=line["smiles"][idx],
                        targets=[line["input_labels"][idx]],
                        features=smiles_to_features[smiles] if features_paths else None,
                        features_generator=features_generator))
                    scores.append(line["input_scores"][idx])
                input_molecules.append(mols)
                input_scores.append(scores)
                output_quantiles.append(
                    np.quantile(line["output_scores"] + [float("inf")],
                                quantile,
                                interpolation="higher"))
        input_scores = torch.FloatTensor(input_scores)
        output_quantiles = torch.FloatTensor(output_quantiles)
        data = MoleculeQuantileDataset(input_molecules, input_scores, output_quantiles, metadata)
        return data

    @staticmethod
    def load_datasets(quantile, file_pattern, features_pattern=None, features_generator=None, num_workers=0):
        worker_fn = functools.partial(
            ChEMBLQuantile._load,
            quantile=quantile,
            features_paths=glob.glob(features_pattern) if features_pattern else None,
            features_generator=features_generator)
        tasks = glob.glob(file_pattern)
        num_workers = min(len(tasks), num_workers)
        if num_workers > 0:
            workers = multiprocessing.Pool(num_workers)
            datasets = workers.map(worker_fn, tasks)
        else:
            datasets = list(map(worker_fn, tasks))
        return datasets

    def prepare_data(self):
        for split in ["train", "val"]:
            datasets = self.load_datasets(
                quantile=self.hparams.quantile,
                file_pattern=getattr(self.hparams, "%s_data" % split),
                features_pattern=getattr(self.hparams, "%s_features" % split),
                features_generator=self.hparams.features_generator,
                num_workers=self.hparams.num_data_workers)
            setattr(self, "%s_datasets" % split, datasets)

    def train_dataloader(self):
        loader = self.quantile_dataloader(
            datasets=self.train_datasets,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            encode_molecules=self.hparams.encode_molecules,
            shuffle=True)
        return loader

    def val_dataloader(self):
        loader = self.quantile_dataloader(
            datasets=self.val_datasets,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            encode_molecules=self.hparams.encode_molecules,
            shuffle=False)
        return loader

    @staticmethod
    def add_task_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        bool = pl.utilities.parsing.str_to_bool

        parser.add_argument("--quantile", type=float, default=0.9)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--max_epochs", type=int, default=20)
        parser.add_argument("--reload_dataloaders_every_epoch", type=bool, default=True)
        parser.add_argument("--default_root_dir", type=str, default="../logs/chembl/quantile/random")

        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_data_workers", type=int, default=10)
        parser.add_argument("--num_support", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--max_train_samples", type=int, default=50000)
        parser.add_argument("--max_val_samples", type=int, default=5000)
        parser.add_argument("--max_test_samples", type=int, default=5000)

        parser.add_argument("--train_data", type=str,
                            default="../data/chembl/random/quantiles/proto/train/k=10/*.json")
        parser.add_argument("--train_features", type=str,
                            default="../data/chembl/random/train/*.npy")
        parser.add_argument("--val_data", type=str,
                            default="../data/chembl/random/quantiles/proto/val/k=10/*.json")
        parser.add_argument("--val_features", type=str,
                            default="../data/chembl/random/val/*.npy")

        parser.add_argument("--encode_molecules", type=bool, default=True)
        parser.add_argument("--warm_start_encoder", type=bool, default=True)
        parser.add_argument("--encoder_ckpt", type=str,
                            default=("../ckpts/chembl/few_shot/random/all_folds/"
                                     "lightning_logs/version_0/checkpoints/epoch=9.ckpt"))
        parser.add_argument("--features_generator", type=str, nargs="+", default=None)
        parser.add_argument("--use_mpn_features", type=bool, default=True)
        parser.add_argument("--use_mol_features", type=bool, default=True)
        parser.add_argument("--mpn_hidden_size", type=int, default=128)
        parser.add_argument("--mol_features_size", type=int, default=200)
        parser.add_argument("--ffnn_hidden_size", type=int, default=128)
        parser.add_argument("--num_ffnn_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--mpn_depth", type=int, default=2)
        parser.add_argument("--undirected_mpn", type=bool, default=False)
        parser.add_argument("--enc_hidden_size", type=int, default=128)

        return parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser
