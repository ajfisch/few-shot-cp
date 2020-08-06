"""Few-Shot learning for ChEMBL."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import nonconformity
import quantile


# ------------------------------------------------------------------------------
#
# Few-shot nonconformity models.
#
# ------------------------------------------------------------------------------


class PairwiseSimilarity(nonconformity.ChEMBLFewShot):
    """Base model for abstract pairwise similarities."""

    def compute_similarity(support_mol_encs, target_mol_encs):
        raise NotImplementedError

    def forward(self, support_encs, query_encs, targets):
        # [query, classes]
        scores = self.compute_similarity(support_encs, query_encs)

        # [batch_size]
        predictions = torch.argmax(scores, dim=1).detach()

        outputs = dict(preds=predictions, targets=targets)

        # Get loss & accuracy.
        if targets is not None:
            outputs["loss"] = F.cross_entropy(scores, targets, reduction="none")
            outputs["acc"] = predictions.eq(targets).float()

        return outputs


class ProtoNet(PairwiseSimilarity):

    def compute_similarity(self, support_encs, query_encs):
        hidden_size = self.hparams.enc_hidden_size
        classes, _, hidden_size = support_encs.size()
        query = query_encs.size(0)

        # [query, classes, hidden_size]
        support_encs = support_encs.mean(dim=1)
        support_encs = support_encs.unsqueeze(0).repeat(query, 1, 1)

        # [query, classes, hidden_size]
        query_encs = query_encs.unsqueeze(1).repeat(1, classes, 1)

        # [query * classes]
        distances_flat = F.pairwise_distance(
            support_encs.view(-1, hidden_size),
            query_encs.view(-1, hidden_size),
            p=2)

        # [query, classes]
        distances = distances_flat.view(-1, classes)

        return -distances


class LRD2(nonconformity.ChEMBLFewShot):
    pass


# ------------------------------------------------------------------------------
#
# Few-shot quantile regression models.
#
# ------------------------------------------------------------------------------


class DeepSet(quantile.ChEMBLQuantile):

    def __init__(self, hparams):
        super(DeepSet, self).__init__(hparams)
        self.embedder = nn.Sequential(
            nn.Linear(1, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU())

        input_size = self.hparams.hidden_size
        if self.hparams.encode_molecules:
            input_size += self.hparams.enc_hidden_size

        self.combiner = nn.Sequential(
            nn.Linear(input_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_size, 1))

    def loss(self, predictions, targets):
        if self.hparams.loss == "quantile":
            above = self.hparams.quantile * F.relu(targets - predictions)
            below = (1 - self.hparams.quantile) * F.relu(predictions - targets)
            loss = above + below
        elif self.hparams.loss == "mse":
            loss = F.mse_loss(predictions, targets, reduction="none")
        else:
            raise ValueError("Invalid loss function.")
        return loss

    def forward(self, mol_encs, scores, targets):
        batch_size = scores.size(0)
        num_support = scores.size(1)

        embeddings = self.embedder(scores.view(-1, 1))
        embeddings = embeddings.view(-1, num_support, self.hparams.hidden_size)
        embeddings = embeddings.sum(dim=1)

        if mol_encs is not None:
            mol_encs = mol_encs.view(batch_size, num_support, -1).sum(dim=1)
            embeddings = torch.cat([embeddings, mol_encs], dim=-1)

        predictions = self.combiner(embeddings).view(-1)

        outputs = dict(preds=predictions.detach(), targets=targets)

        if targets is not None:
            outputs["loss"] = self.loss(predictions, targets)

        return outputs

    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.add_argument("--hidden_size", type=int, default=128)
        parser.add_argument("--loss", type=str, choices=["quantile", "mse"], default="mse")
        return parser
