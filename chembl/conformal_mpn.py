"""Conformal MPN pre-training for getting nonconformity scores."""

import argparse
import chemprop
import collections
import csv
import encoders
import numpy as np
import os
import pytorch_lightning as pl
import shutil
import subprocess
import torch
import torch.optim as optim
import tqdm


def construct_molecule_batch(batch):
    smiles = [x.smiles for x in batch]
    tasks = [x.task for x in batch]
    inputs = chemprop.data.data.construct_molecule_batch(batch)

    mol_graph = inputs.batch_graph()
    if mol_graph:
        mol_graph = mol_graph.get_components()

    mol_features = inputs.features()
    if mol_features:
        mol_features = torch.from_numpy(np.stack(mol_features)).float()

    targets = inputs.targets()
    targets = [t[0] for t in targets]
    targets = torch.Tensor(targets).float()

    inputs = [mol_graph, mol_features, targets]
    return inputs, (smiles, tasks)


class FewShotSampler(torch.utils.data.Sampler):
    """Data sampler for few-shot regression on ChEMBL.

    Randomly samples batches of support examples + query examples.
    """

    def __init__(self, indices, tasks_per_batch, num_support, num_query, iters=None):
        # Indices is a list of lists.
        self.indices = indices
        self.tasks_per_batch = tasks_per_batch
        self.num_support = num_support
        self.num_query = num_query
        self.iters = iters or float("inf")

    def create_batch(self):
        batch_indices = []
        for _ in range(self.tasks_per_batch):
            task_idx = np.random.randint(0, len(self.indices))
            task_indices = self.indices[task_idx]
            example_indices = np.random.choice(
                task_indices,
                size=self.num_support + self.num_query,
                replace=False)
            batch_indices.extend(example_indices)

    def __iter__(self):
        i = 0
        while i < self.iters:
            yield self.create_batch()
            i += 1

    def __len__(self):
        return self.iters


class ConformalMPN(pl.LightningModule):
    """Few-shot MPN to compute regression scores."""

    def __init__(self, hparams):
        super(ConformalMPN, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.encoder = encoders.MoleculeEncoder(hparams)
        self.head = encoders.R2D2Head(hparams)

    def encode(self, inputs, sizes):
        tasks_per_batch, n_support, n_query = sizes
        mol_graph, mol_features, targets = inputs

        # [tasks_per_batch * (n_support + n_query), dim]
        mol_encs = self.encoder(mol_graph, mol_features)

        # [tasks_per_batch, n_support + n_query, dim]
        mol_encs = mol_encs.view(tasks_per_batch, n_support + n_query, -1)
        targets = targets.view(tasks_per_batch, n_support + n_query)

        # [tasks_per_batch, n_support, dim]
        support = mol_encs[:, :n_support, :]
        support_targets = targets[:, :n_support]

        # [tasks_per_batch, n_query, dim]
        query = mol_encs[:, n_support:, :]
        query_targets = targets[:, n_support:]

        return (query, support, support_targets), query_targets

    def forward(self, inputs, sizes):
        inputs, targets = self.encode(inputs, sizes)

        # [tasks_per_batch, n_query]
        y_pred = self.head(*inputs)

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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-6)
        return [optimizer], [scheduler]

    @staticmethod
    def load_dataset(dataset_file, features_file=None, features_generator=None):
        if features_file is not None:
            smiles_to_idx, idx_to_features = np.load(features_file, allow_pickle=True)

        # Read dataset.
        mols = []
        num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
        with open(dataset_file, "r") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            for row in tqdm.tqdm(reader, total=num_lines - 1, desc="reading dataset"):
                smiles = row[columns[0]]
                task = row[columns[1]]
                target = row[columns[2]]

                # Slight misuse of the MoleculeDatapoint to include single target + task id.
                mol = chemprop.data.MoleculeDatapoint(
                    smiles=smiles,
                    targets=[target],
                    features=idx_to_features[smiles_to_idx[smiles]].copy() if features_file else None,
                    features_generator=features_generator)
                mol.task = task

                mols.append(mol)

        # Transform to dataset.
        dataset = chemprop.data.MoleculeDataset(mols)

        # Gather task examples.
        tasks = {}
        indices = collections.defaultdict(list)
        for idx, mol in enumerate(dataset):
            if mol.task not in tasks:
                tasks[mol.task] = len(tasks)
            indices[tasks[mol.task]].append(idx)

        return dataset, indices, tasks

    def prepare_data(self):
        for split in ["train", "val"]:
            dataset, indices, tasks = self.load_dataset(
                dataset_file=getattr(self.hparams, "%s_data" % split),
                features_file=getattr(self.hparams, "%s_features" % split),
                features_generator=self.hparams.features_generator,
                limit=None if split == "train" else self.hparams.subsample_val)
            setattr(self, "%s_dataset" % split, dataset)
            setattr(self, "%s_indices" % split, dataset)
            setattr(self, "%s_tasks" % split, dataset)

    def train_dataloader(self):
        sampler = FewShotSampler(
            indices=self.train_indices,
            tasks_per_batch=self.hparams.tasks_per_batch,
            num_support=self.hparams.num_support,
            num_query=self.hparams.num_query,
            iters=self.hparams.max_train_iters)
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_molecule_batch)
        return loader

    def val_dataloader(self):
        sampler = FewShotSampler(
            indices=self.val_indices,
            tasks_per_batch=self.hparams.tasks_per_batch,
            num_support=self.hparams.num_support,
            num_query=self.hparams.num_query,
            iters=self.hparams.max_val_iters)
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_sampler=sampler,
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
        parser.add_argument("--checkpoint_dir", type=str, default="../ckpts/chembl/conformal_mpn")
        parser.add_argument("--overwrite", type=bool, default=True)

        parser.add_argument("--tasks_per_batch", type=int, default=4)
        parser.add_argument("--num_support", type=int, default=20)
        parser.add_argument("--num_query", type=int, default=16)
        parser.add_argument("--max_train_iters", type=int, default=2000)
        parser.add_argument("--max_val_iters", type=int, default=100)
        parser.add_argument("--max_epochs", type=int, default=40)

        parser.add_argument("--num_data_workers", type=int, default=20)
        parser.add_argument("--train_data", type=str, default="../data/chembl/train_molecules.csv")
        parser.add_argument("--train_features", type=str, default="../data/chembl/features/train_molecules.npy")
        parser.add_argument("--val_data", type=str, default="../data/chembl/val_molecules.csv")
        parser.add_argument("--val_features", type=str, default="../data/chembl/features/val_molecules.npy")

        parser.add_argument("--features_generator", type=str, nargs="+", default=None)
        parser.add_argument("--use_mpn_features", type="bool", default=True)
        parser.add_argument("--use_mol_features", type="bool", default=True)
        parser.add_argument("--mpn_hidden_size", type=int, default=128)
        parser.add_argument("--mol_features_size", type=int, default=200)
        parser.add_argument("--ffnn_hidden_size", type=int, default=128)
        parser.add_argument("--num_ffnn_layers", type=int, default=3)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--mpn_depth", type=int, default=2)
        parser.add_argument("--undirected_mpn", type="bool", default=False)
        parser.add_argument("--enc_hidden_size", type=int, default=128)

        return parser


def main(args):
    pl.seed_everything(args.seed)
    model = ConformalMPN(hparams=args)
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
    parser = ConformalMPN.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
