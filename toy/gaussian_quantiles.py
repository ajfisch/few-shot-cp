"""Approximate quantiles of few-shot gaussian samples."""

import argparse
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning


def create_tasks(n):
    mu = np.random.uniform(0, 1, size=n)
    std = np.random.uniform(0, 1, size=n)
    tasks = list(zip(mu, std))
    return tasks


class DeepSet(pytorch_lightning.LightningModule):

    def __init__(self, hparams):
        super(DeepSet, self).__init__()
        self.hparams = hparams
        self.embedder = nn.Sequential(
            nn.Linear(1, hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.hidden_size, hparams.hidden_size),
            nn.ReLU())
        factor = 3 if hparams.pool == "all" else 1
        self.combiner = nn.Sequential(
            nn.Linear(hparams.hidden_size * factor, hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.hidden_size, hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.hidden_size, 1))

    def forward(self, x):
        embeddings = self.embedder(x.view(-1, 1))
        embeddings = embeddings.view(-1, self.hparams.k_shot, self.hparams.hidden_size)
        if self.hparams.pool == "sum":
            embeddings = embeddings.sum(dim=1)
        elif self.hparams.pool == "max":
            embeddings = embeddings.max(dim=1)[0]
        elif self.hparams.pool == "min":
            embeddings = embeddings.min(dim=1)[0]
        else:
            embeddings = torch.cat(
                [embeddings.sum(dim=1),
                 embeddings.max(dim=1)[0],
                 embeddings.min(dim=1)[0]],
                dim=1)
        outputs = self.combiner(embeddings)
        return outputs.view(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y, y_hat)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = (y - y_hat).abs()
        baseline_loss = y.clone()
        for i in range(len(x)):
            y_emp = np.quantile(x[i].cpu().numpy(), self.hparams.alpha,
                                interpolation="higher")
            baseline_loss[i] = (y[i] - y_emp).abs()
        return {"val_loss": loss, "baseline_val_loss": baseline_loss}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["val_loss"] for x in outputs])
        baseline_loss = torch.cat([x["baseline_val_loss"] for x in outputs])
        tensorboard_logs = {"avg_val_loss": loss.mean(), "avg_baseline_val_loss": baseline_loss.mean(),
                            "std_val_loss": loss.std(), "std_baseline_val_loss": baseline_loss.std()}
        return {"val_loss": loss.mean(), "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = (y - y_hat).abs()
        baseline_loss = y.clone()
        for i in range(len(x)):
            y_emp = np.quantile(x[i].cpu().numpy(), self.hparams.alpha,
                                interpolation="higher")
            baseline_loss[i] = (y[i] - y_emp).abs()
        return {"test_loss": loss, "baseline_test_loss": baseline_loss}

    def test_epoch_end(self, outputs):
        loss = torch.cat([x["test_loss"] for x in outputs])
        baseline_loss = torch.cat([x["baseline_test_loss"] for x in outputs])
        tensorboard_logs = {"avg_test_loss": loss.mean(), "avg_baseline_test_loss": baseline_loss.mean(),
                            "std_test_loss": loss.std(), "std_baseline_test_loss": baseline_loss.std()}
        return {"test_loss": loss.mean(), "log": tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        num_tasks = self.hparams.num_train_tasks
        num_examples = self.hparams.num_train_examples
        k = self.hparams.k_shot
        tasks = create_tasks(num_tasks)
        inputs = np.empty((num_tasks * num_examples, k))
        targets = np.empty((num_tasks * num_examples))
        i = 0
        for mu, std in tasks:
            for _ in range(num_examples):
                inputs[i] = np.random.normal(mu, std, size=k)
                targets[i] = scipy.stats.norm.ppf(self.hparams.alpha, mu, std)
                i += 1
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=self.hparams.batch_size)
        return loader

    def val_dataloader(self):
        num_tasks = self.hparams.num_val_tasks
        num_examples = self.hparams.num_val_examples
        k = self.hparams.k_shot
        tasks = create_tasks(num_tasks)
        inputs = np.empty((num_tasks * num_examples, k))
        targets = np.empty((num_tasks * num_examples))
        i = 0
        for mu, std in tasks:
            for _ in range(num_examples):
                inputs[i] = np.random.normal(mu, std, size=k)
                targets[i] = scipy.stats.norm.ppf(self.hparams.alpha, mu, std)
                i += 1
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size)
        return loader

    def test_dataloader(self):
        num_tasks = self.hparams.num_test_tasks
        num_examples = self.hparams.num_test_examples
        k = self.hparams.k_shot
        tasks = create_tasks(num_tasks)
        inputs = np.empty((num_tasks * num_examples, k))
        targets = np.empty((num_tasks * num_examples))
        i = 0
        for mu, std in tasks:
            for _ in range(num_examples):
                inputs[i] = np.random.normal(mu, std, size=k)
                targets[i] = scipy.stats.norm.ppf(self.hparams.alpha, mu, std)
                i += 1
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size)
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=0.001, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--hidden_size", default=32, type=int)
        parser.add_argument("--pool", default="all", type=str,
                            choices=["sum", "max", "min", "both"])
        parser.add_argument("--alpha", default=0.9, type=float)

        parser.add_argument("--k_shot", default=100, type=int)
        parser.add_argument("--num_train_tasks", default=100, type=int)
        parser.add_argument("--num_train_examples", default=1000, type=int)
        parser.add_argument("--num_val_tasks", default=25, type=int)
        parser.add_argument("--num_val_examples", default=100, type=int)
        parser.add_argument("--num_test_tasks", default=25, type=int)
        parser.add_argument("--num_test_examples", default=100, type=int)

        return parser


def main(args):
    pytorch_lightning.seed_everything(42)
    model = DeepSet(hparams=args)
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser = DeepSet.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
