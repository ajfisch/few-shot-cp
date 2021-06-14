"""Quantile prediction using a deep set NN."""

import argparse
import collections
import json
import numpy as np
import os
import pytorch_lightning as pl
import shutil
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

ExampleSet = collections.namedtuple(
    "ExampleSet", ["scores", "target", "task", "f_probs"])


class DeepSet(nn.Module):
    """Set --> scalar."""

    def __init__(self, hparams):
        super(DeepSet, self).__init__()
        input_size = 1

        self.embedder = nn.Sequential(
            nn.Linear(input_size, hparams.set_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.set_hidden_size, hparams.set_hidden_size),
            nn.ReLU())

        self.combiner = nn.Sequential(
            nn.Linear(hparams.set_hidden_size, hparams.set_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.set_hidden_size, hparams.set_hidden_size),
            nn.ReLU(),
            nn.Linear(hparams.set_hidden_size, 1))

    def forward(self, inputs):
        """Encode set inputs.

        inputs: <float> [batch_size, n_support, dim]
        """
        # [batch_size, n_support, set_hidden_size]
        set_embs = self.embedder(inputs)

        # [batch_size, set_hidden_size]
        set_embs = set_embs.mean(dim=1)

        # [batch_size]
        preds = self.combiner(set_embs).view(-1)

        return preds


def construct_batch(batch):
    tasks = [x.task for x in batch]

    scores = torch.Tensor([x.scores for x in batch]).float()
    targets = torch.Tensor([x.target for x in batch]).float()
    f_probs = torch.Tensor([x.f_probs for x in batch]).float()

    inputs = [scores, targets]
    return inputs, tasks, f_probs


class QuantileSNN(pl.LightningModule):
    """Few-shot DeepSet to compute quantiles."""

    def __init__(self, hparams):
        super(QuantileSNN, self).__init__()
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams
        self.head = DeepSet(hparams)

    def predict(self, scores):
        # [batch_size, n_support, 1]
        input_encs = scores.unsqueeze(-1)

        # [batch_size]
        y_pred = self.head(input_encs)

        return y_pred

    def forward(self, inputs):
        targets = inputs[1]

        y_pred = self.predict(
            scores=inputs[0])

        loss = (targets - y_pred).pow(2)

        return dict(loss=loss, preds=y_pred)

    def training_step(self, batch, batch_idx):
        inputs, _, _ = batch
        outputs = self.forward(inputs)
        loss = outputs["loss"].mean()
        tensorboard_logs = {"train_loss": loss.detach()}
        self.log("loss", loss)
        #return {"loss": loss, "log": tensorboard_logs}
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, _, _ = batch
        return self.forward(inputs)

    def test_step(self, batch, batch_idx):
        inputs, _, f_probs = batch
        return self.forward(inputs), f_probs

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"].detach() for x in outputs], dim=0).mean()
        tensorboard_logs = {"val_loss": loss}
        tqdm_dict = tensorboard_logs
        #return {"val_loss": loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}
        self.log("val_loss", loss)
        return

    def test_epoch_end(self, results):
        # results: [(outputs, f_probs)] * batches
        output_path = os.path.join(self.hparams.checkpoint_dir, "predictions.jsonl")
        with open(output_path, "w") as f:
            for outputs, f_probs in results:
                preds = outputs['preds'].tolist()
                probs = f_probs.tolist()
                for pred, input_probs in zip(preds, probs):
                    f.write(json.dumps({'pred': pred, 'scores': input_probs}) + '\n')

        loss = torch.cat([x[0]["loss"].detach() for x in results], dim=0).mean()
        tensorboard_logs = {"test_loss": loss}
        tqdm_dict = tensorboard_logs
        #return {"val_loss": loss, "progress_bar": tqdm_dict, "log": tensorboard_logs}
        self.log("test_loss", loss)
        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=0.1, patience=2, min_lr=1e-6),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def load_dataset(dataset_file, alpha):
        dataset = []
        num_lines = int(subprocess.check_output(["wc", "-l", dataset_file], encoding="utf8").split()[0])
        with open(dataset_file, "r") as f:
            for line in tqdm.tqdm(f, total=num_lines):
                line = json.loads(line)

                #scores = line["s_scores"]
                #target = np.quantile(line["f_scores"], alpha, interpolation="higher")
                scores = [-x for x in  line["s_probs"]]
                target = np.quantile([-x for x in line["f_probs"]], alpha, interpolation="higher")
                task = line["task"]
                f_probs = [-x for x in line["f_probs"]]
                dataset.append(ExampleSet(scores, target, task, f_probs))

        return dataset

    def prepare_data(self):
        for split in ["train", "val"]:
            dataset = self.load_dataset(
                dataset_file=getattr(self.hparams, "%s_data" % split),
                alpha=self.hparams.alpha)
            setattr(self, "%s_dataset" % split, dataset)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_batch)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_data_workers,
            collate_fn=construct_batch)
        return loader

    def test_dataloader(self):
        return self.val_dataloader()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.register("type", "bool", pl.utilities.parsing.str_to_bool)

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--gpus", type=int, nargs="+", default=None)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--checkpoint_dir", type=str, default="results/10_way/quantile_snn/q=0.80")
        parser.add_argument("--overwrite", type="bool", default=False)
        parser.add_argument("--only_test", type="bool", default=False)

        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--max_epochs", type=int, default=15)

        parser.add_argument("--num_data_workers", type=int, default=20)
        parser.add_argument("--train_data", type=str, default="results/10_way/train_quantile.jsonl")
        parser.add_argument("--val_data", type=str, default="results/10_way/val_quantile.jsonl")

        parser.add_argument("--alpha", type=float, default=0.8)
        parser.add_argument("--set_hidden_size", type=int, default=64)

        return parser


def main(args):
    pl.seed_everything(args.seed)
    model = QuantileSNN(hparams=args)
    print(model)
    if os.path.exists(args.checkpoint_dir) and os.listdir(args.checkpoint_dir) and not args.only_test:
        if not args.overwrite:
            raise RuntimeError("Experiment directory is not empty.")
        else:
            shutil.rmtree(args.checkpoint_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.checkpoint_dir, "weights.pt"),
        save_top_k=1,
        monitor="val_loss",
        mode="min")
    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir=os.path.join(args.checkpoint_dir, "logs"))
    logger_csv = pl.loggers.CSVLogger(
        save_dir=os.path.join(args.checkpoint_dir, "logs"))
    trainer = pl.Trainer(
        logger=[logger, logger_csv],
        callbacks=[checkpoint_callback],
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        terminate_on_nan=True)

    if not args.only_test:
        trainer.fit(model)

    ckpt = os.path.join(args.checkpoint_dir, "weights.pt.ckpt")
    trainer.test(
        model=QuantileSNN.load_from_checkpoint(ckpt),
        verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = QuantileSNN.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
