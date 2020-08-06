"""Few-Shot learning for ChEMBL."""

import argparse
import csv
import functools
import glob
import multiprocessing
import os

import chemprop
import molecules
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim


def construct_molecule_batch(batch):
    smiles = [x.smiles for x in batch]
    tasks = [x.task for x in batch]
    assert(len(set(tasks)) == 1)
    task = tasks[0]
    inputs = chemprop.data.data.construct_molecule_batch(batch)
    mol_graph = inputs.batch_graph()
    if mol_graph:
        mol_graph = mol_graph.get_components()
    mol_features = inputs.features()
    if mol_features:
        mol_features = torch.from_numpy(np.stack(mol_features)).float()
    targets = inputs.targets()
    if targets:
        targets = torch.Tensor(targets).view(-1).long()
    inputs = [mol_graph, mol_features, targets]
    return inputs, (smiles, task)


class FewShotSampler(torch.utils.data.Sampler):

    def __init__(self, task_offsets, num_support, num_query, max_examples=None):
        task_sizes = []
        for i in range(len(task_offsets)):
            if i == 0:
                size = task_offsets[i]
            else:
                size = task_offsets[i] - task_offsets[i - 1]
            if size % 2 != 0:
                raise RuntimeError("Expected even sized tasks.")
            task_sizes.append(size)
        if num_query % 2 != 0:
            raise RuntimeError("Expected even sized queries.")
        self.task_sizes = task_sizes
        self.task_offsets = task_offsets
        self.num_support = num_support
        self.num_query = num_query
        self.max_examples = max_examples or float("inf")

    def create_example(self):
        example_indices = []

        # Sample task.
        task_idx = np.random.randint(0, len(self.task_sizes))

        # Set test examples.
        tests = []

        # Sample negative support.
        start_idx = 0 if task_idx == 0 else self.task_offsets[task_idx - 1]
        end_idx = start_idx + self.task_sizes[task_idx] // 2
        negatives = np.random.choice(
            range(start_idx, end_idx),
            size=self.num_support,
            replace=False)
        example_indices.extend(negatives)

        # Sample negative queries.
        while len(tests) < self.num_query // 2:
            test_idx = np.random.randint(start_idx, end_idx)
            while test_idx in negatives:
                test_idx = np.random.randint(start_idx, end_idx)
                tests.append(test_idx)

        # Sample positive support
        start_idx = end_idx
        end_idx = self.task_offsets[task_idx]
        positives = np.random.choice(
            range(start_idx, end_idx),
            size=self.num_support,
            replace=False)
        example_indices.extend(positives)

        # Sample positive queries.
        while len(tests) < self.num_query:
            test_idx = np.random.randint(start_idx, end_idx)
            while test_idx in positives:
                test_idx = np.random.randint(start_idx, end_idx)
                tests.append(test_idx)
        example_indices.extend(tests)

        return example_indices

    def __iter__(self):
        examples = 0
        while examples < self.max_examples:
            yield self.create_example()
            examples += 1

    def __len__(self):
        return self.max_examples


class ChEMBLFewShot(pl.LightningModule):

    def __init__(self, hparams):
        super(ChEMBLFewShot, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.encoder = molecules.MoleculeEncoder(hparams)

    def get_few_shot(self, inputs):
        mol_graph, mol_features, targets = inputs
        mol_encs = self.encoder(mol_graph, mol_features)

        classes = self.hparams.num_classes
        support = self.hparams.num_support

        support_encs = mol_encs[:classes * support]
        support_encs = support_encs.view(classes, support, -1)
        query_encs = mol_encs[classes * support:]

        targets = targets[classes * support:]

        return self.forward(support_encs, query_encs, targets)

    def get_few_shot_calibration(self, inputs):
        mol_graph, mol_features, targets = inputs
        mol_encs = self.encoder(mol_graph, mol_features)

        classes = self.hparams.num_classes
        support = self.hparams.num_support

        support_encs = mol_encs[:classes * support]
        support_encs = support_encs.view(classes, support, -1)
        support_targets = targets[:classes * support].view(classes, support)
        query_encs = mol_encs[classes * support:]
        query_targets = targets[classes * support:]

        outputs = []
        for i in range(support):
            idx = [j != i for j in range(support)]
            support_encs_cal = support_encs[:, idx]
            query_encs_cal = torch.cat([support_encs[:, i], query_encs], dim=0)
            targets_cal = torch.cat([support_targets[:, i], query_targets], dim=0)
            output = self.forward(support_encs_cal, query_encs_cal, targets_cal)
            output = dict(
                loss=output["loss"][:classes].unsqueeze(0),
                targets=output["targets"][:classes].unsqueeze(0),
                cv_loss=output["loss"][classes:].unsqueeze(0),
                cv_targets=output["targets"][classes:].unsqueeze(0))
            outputs.append(output)

        calibration = {}
        for k in outputs[0].keys():
            calibration[k] = torch.cat([x[k] for x in outputs])

        return calibration

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        outputs = self.get_few_shot(inputs)
        loss = outputs["loss"].mean()
        tensorboard_logs = {"train_loss": loss.detach()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        inputs, _ = batch
        return self.get_few_shot(inputs)

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
        inputs, _ = batch
        return self.get_few_shot(inputs)

    def test_epoch_end(self, outputs):
        loss = torch.cat([x["loss"] for x in outputs])
        acc = torch.cat([x["acc"] for x in outputs])
        tensorboard_logs = {"test_loss": loss.mean().item(),
                            "test_acc": acc.mean().item()}
        tqdm_dict = tensorboard_logs
        return {"progress_bar": tqdm_dict, "log": tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def few_shot_dataloader(datasets, num_query, num_support, num_workers, max_examples=None):
        dataset = torch.utils.data.ConcatDataset(datasets)
        sampler = FewShotSampler(
            task_offsets=dataset.cumulative_sizes,
            num_query=num_query,
            num_support=num_support,
            max_examples=max_examples)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
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
            task = columns[1]
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

            data = []
            for smiles, target in negatives + positives:
                datapoint = chemprop.data.MoleculeDatapoint(
                    smiles=smiles,
                    targets=[target],
                    features=idx_to_features[smiles_to_idx[smiles]] if features_paths else None,
                    features_generator=features_generator)
                datapoint.task = task
                data.append(datapoint)
            data = chemprop.data.MoleculeDataset(data)

        return data

    @staticmethod
    def load_datasets(file_pattern, features_pattern=None, features_generator=None, num_workers=0):
        worker_fn = functools.partial(
            ChEMBLFewShot._load,
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
        for split in ["train", "val", "test"]:
            datasets = self.load_datasets(
                file_pattern=getattr(self.hparams, "%s_data" % split),
                features_pattern=getattr(self.hparams, "%s_features" % split),
                features_generator=self.hparams.features_generator,
                num_workers=self.hparams.num_data_workers)
            setattr(self, "%s_datasets" % split, datasets)

    def train_dataloader(self):
        loader = self.few_shot_dataloader(
            datasets=self.train_datasets,
            num_query=self.hparams.train_num_query,
            num_support=self.hparams.num_support,
            num_workers=self.hparams.num_data_workers,
            max_examples=self.hparams.max_train_samples)
        return loader

    def val_dataloader(self):
        loader = self.few_shot_dataloader(
            datasets=self.val_datasets,
            num_query=self.hparams.test_num_query,
            num_support=self.hparams.num_support,
            num_workers=self.hparams.num_data_workers,
            max_examples=self.hparams.max_val_samples)
        return loader

    def test_dataloader(self):
        loader = self.few_shot_dataloader(
            datasets=self.test_datasets,
            num_query=self.hparams.test_num_query,
            num_support=self.hparams.num_support,
            num_workers=self.hparams.num_data_workers,
            max_examples=self.hparams.max_test_samples)
        return loader

    @staticmethod
    def add_task_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.register("type", "bool", pl.utilities.parsing.str_to_bool)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--max_epochs", type=int, default=10)
        parser.add_argument("--reload_dataloaders_every_epoch", type="bool", default=True)
        parser.add_argument("--default_root_dir", type=str, default="../logs/chembl/few_shot/random")

        parser.add_argument("--num_data_workers", type=int, default=10)
        parser.add_argument("--num_support", type=int, default=10)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--train_num_query", type=int, default=32)
        parser.add_argument("--test_num_query", type=int, default=32)

        parser.add_argument("--max_train_samples", type=int, default=50000)
        parser.add_argument("--max_val_samples", type=int, default=5000)
        parser.add_argument("--max_test_samples", type=int, default=5000)

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
