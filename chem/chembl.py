"""Few-Shot learning for ChEMBL."""

import argparse
import csv
import functools
import glob
import math
import multiprocessing
import os

import chemprop
import molecules
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim


def construct_molecule_batch(batch):
    batch = chemprop.data.data.construct_molecule_batch(batch)
    mol_graph = batch.batch_graph()
    if mol_graph:
        mol_graph = mol_graph.get_components()
    mol_features = batch.features()
    if mol_features:
        mol_features = torch.from_numpy(np.stack(mol_features)).float()
    targets = batch.targets()
    if targets:
        targets = torch.Tensor(targets).view(-1).long()
    batch = [mol_graph, mol_features, targets]
    return batch


class FewShotSampler(torch.utils.data.Sampler):

    def __init__(self, task_offsets, batch_size, support_size, max_examples=None):
        task_sizes = []
        for i in range(len(task_offsets)):
            if i == 0:
                size = task_offsets[i]
            else:
                size = task_offsets[i] - task_offsets[i - 1]
            if size % 2 != 0:
                raise RuntimeError("Expected even sized tasks.")
            task_sizes.append(size)
        self.task_sizes = task_sizes
        self.task_offsets = task_offsets
        self.batch_size = batch_size
        self.support_size = support_size
        self.max_examples = max_examples or float("inf")

    def create_example(self):
        example_indices = []

        # Sample task.
        task_idx = np.random.randint(0, len(self.task_sizes))

        # Sample negative examples.
        start_idx = 0 if task_idx == 0 else self.task_offsets[task_idx - 1]
        end_idx = start_idx + self.task_sizes[task_idx] // 2
        negatives = np.random.choice(
            range(start_idx, end_idx),
            size=self.support_size,
            replace=False)
        example_indices.extend(negatives)

        # Sample positive examples.
        start_idx = end_idx
        end_idx = self.task_offsets[task_idx]
        positives = np.random.choice(
            range(start_idx, end_idx),
            size=self.support_size,
            replace=False)
        example_indices.extend(positives)

        # Sample test example.
        start_idx = 0 if task_idx == 0 else self.task_offsets[task_idx - 1]
        end_idx = self.task_offsets[task_idx]
        support = set(positives.tolist() + negatives.tolist())
        test_idx = np.random.randint(start_idx, end_idx)
        while test_idx in support:
            test_idx = np.random.randint(start_idx, end_idx)
        example_indices.append(test_idx)

        return example_indices

    def __iter__(self):
        examples = 0
        while examples < self.max_examples:
            batch_size = min(self.batch_size, self.max_examples - examples)
            batch_indices = []
            for _ in range(batch_size):
                batch_indices.extend(self.create_example())
            yield batch_indices
            examples += batch_size

    def __len__(self):
        return math.ceil(self.max_examples / self.batch_size)


class ChEMBLFewShot(pl.LightningModule):

    def __init__(self, hparams):
        super(ChEMBLFewShot, self).__init__()
        self.hparams = hparams
        self.encoder = molecules.MoleculeEncoder(hparams)

    def get_outputs(self, inputs):
        return self.forward(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.get_outputs(batch)
        loss = outputs["loss"].mean()
        tensorboard_logs = {"train_loss": loss.detach()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        return self.get_outputs(batch)

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"] for x in outputs]).mean().item()
        acc = torch.cat([x["acc"] for x in outputs]).mean().item()
        tensorboard_logs = {"val_loss": loss,
                            "val_acc": acc}
        tqdm_dict = tensorboard_logs
        return {"val_loss": loss,
                "val_acc": acc,
                "progress_bar": tqdm_dict,
                "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        return self.get_outputs(batch)

    def test_epoch_end(self, outputs):
        loss = torch.cat([x["loss"] for x in outputs])
        acc = torch.cat([x["acc"] for x in outputs])
        tensorboard_logs = {"test_loss": loss.mean().item(),
                            "test_acc": acc.mean().item()}
        tqdm_dict = tensorboard_logs
        return {"progress_bar": tqdm_dict, "log": tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def few_shot_dataloader(self, datasets, batch_size, max_examples=None):
        dataset = torch.utils.data.ConcatDataset(datasets)
        sampler = FewShotSampler(
            task_offsets=dataset.cumulative_sizes,
            batch_size=batch_size,
            support_size=self.hparams.k_shot,
            max_examples=max_examples)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_molecule_batch)
        return loader

    @staticmethod
    def _load(path, features_paths=None, features_generator=None):
        if features_paths is not None:
            dataset = os.path.splitext(os.path.basename(path))[0]
            matching = None
            for features_path in features_paths:
                key = os.path.splitext(os.path.basename(features_path))[0]
                if key == dataset:
                    matching = features_path
                    break
            if not matching:
                raise RuntimeError("Did not find features file for %s" % dataset)
            smiles_to_idx, idx_to_features = np.load(matching, allow_pickle=True)

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            positives = []
            negatives = []
            for row in reader:
                smiles = row[columns[0]]
                target = float(row[columns[1]])
                if target:
                    positives.append((smiles, target))
                else:
                    negatives.append((smiles, target))

            if len(positives) != len(negatives):
                raise RuntimeError("Assumed balanced classes. Positives not equal to negatives.")

            data = chemprop.data.MoleculeDataset([
                chemprop.data.MoleculeDatapoint(
                    smiles=smiles,
                    targets=[target],
                    features=idx_to_features[smiles_to_idx[smiles]] if features_paths else None,
                    features_generator=features_generator)
                for smiles, target in negatives + positives])

        return data

    def load_datasets(self, file_pattern, features_pattern=None):
        worker_fn = functools.partial(
            self._load,
            features_paths=glob.glob(features_pattern) if features_pattern else None,
            features_generator=self.hparams.features_generator)
        tasks = glob.glob(file_pattern)
        num_workers = min(len(tasks), self.hparams.num_data_workers)
        if num_workers > 0:
            workers = multiprocessing.Pool(num_workers)
            datasets = workers.map(worker_fn, tasks)
        else:
            datasets = list(map(worker_fn, tasks))
        return datasets

    def prepare_data(self):
        for split in ["train", "val", "test"]:
            datasets = self.load_datasets(
                file_pattern=getattr(self.hparams, "%s_data" % split),
                features_pattern=getattr(self.hparams, "%s_features" % split))
            setattr(self, "%s_datasets" % split, datasets)

    def train_dataloader(self):
        loader = self.few_shot_dataloader(
            datasets=self.train_datasets,
            batch_size=self.hparams.batch_size,
            max_examples=self.hparams.max_train_samples)
        return loader

    def val_dataloader(self):
        loader = self.few_shot_dataloader(
            datasets=self.val_datasets,
            batch_size=self.hparams.test_batch_size,
            max_examples=self.hparams.max_val_samples)
        return loader

    def test_dataloader(self):
        loader = self.few_shot_dataloader(
            datasets=self.test_datasets,
            batch_size=self.hparams.test_batch_size,
            max_examples=self.hparams.max_test_samples)
        return loader

    @staticmethod
    def add_task_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.register("type", "bool", pl.utilities.parsing.str_to_bool)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--max_epochs", type=int, default=20)
        parser.add_argument("--reload_dataloaders_every_epoch", type="bool", default=True)
        parser.add_argument("--default_root_dir", type=str, default="../logs/chembl/few_shot/random")

        parser.add_argument("--num_data_workers", type=int, default=10)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--test_batch_size", type=int, default=20)
        parser.add_argument("--k_shot", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=2)

        parser.add_argument("--max_train_samples", type=int, default=5000)
        parser.add_argument("--max_val_samples", type=int, default=50)
        parser.add_argument("--max_test_samples", type=int, default=50)

        parser.add_argument("--train_data", type=str, default="../data/chembl/random/train/*.csv")
        parser.add_argument("--train_features", type=str, default="../data/chembl/random/train/*.npy")
        parser.add_argument("--val_data", type=str, default="../data/chembl/random/val/*.csv")
        parser.add_argument("--val_features", type=str, default="../data/chembl/random/val/*.npy")
        parser.add_argument("--test_data", type=str, default="../data/chembl/random/test/*.csv")
        parser.add_argument("--test_features", type=str, default="../data/chembl/random/test/*.npy")

        parser.add_argument("--features_generator", type=str, nargs="+", default=None)
        parser.add_argument("--use_mpn_features", type="bool", default=True)
        parser.add_argument("--use_mol_features", type="bool", default=True)
        parser.add_argument("--mpn_hidden_size", type=int, default=128)
        parser.add_argument("--mol_features_size", type=int, default=200)
        parser.add_argument("--ffnn_hidden_size", type=int, default=128)
        parser.add_argument("--num_ffnn_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--mpn_depth", type=int, default=2)
        parser.add_argument("--undirected_mpn", type="bool", default=False)
        parser.add_argument("--enc_hidden_size", type=int, default=128)

        return parser

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser
