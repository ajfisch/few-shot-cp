"""Few-Shot learning for ChEMBL."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chembl
import higher


class PairwiseSimilarity(chembl.ChEMBLFewShot):
    """Base model for abstract pairwise similarities."""

    def compute_similarity(support_mol_encs, target_mol_encs):
        raise NotImplementedError

    def forward(self, inputs):
        mol_graph, mol_features, targets = inputs

        # [batch_size * (classes * k + 1), hidden_size]
        mol_encs = self.encoder(mol_graph, mol_features)

        # [batch_size, classes * k + 1, hidden_size]
        hidden_size = self.hparams.enc_hidden_size
        k = self.hparams.k_shot
        classes = self.hparams.num_classes
        mol_encs = mol_encs.view(-1, classes * k + 1, hidden_size)

        # [batch_size, classes, k, hidden_size]
        support_mol_encs = mol_encs[:, :-1, :].view(-1, classes, k, hidden_size)

        # [batch_size, hidden_size]
        target_mol_encs = mol_encs[:, -1, :]

        # [batch_size, classes]
        scores = self.compute_similarity(support_mol_encs, target_mol_encs)

        # [batch_size]
        targets = targets.view(-1, classes * k + 1)[:, -1]

        # [batch_size]
        predictions = torch.argmax(scores, dim=1).detach()

        outputs = dict(preds=predictions)

        # Get loss & accuracy.
        if targets is not None:
            outputs["loss"] = F.cross_entropy(scores, targets, reduction="none")
            outputs["acc"] = predictions.eq(targets).float()

        return outputs


class ProtoNet(PairwiseSimilarity):

    def compute_similarity(self, support_mol_encs, target_mol_encs):
        hidden_size = self.hparams.enc_hidden_size
        classes = self.hparams.num_classes

        # [batch_size, classes, hidden_size]
        support_mol_encs = support_mol_encs.mean(dim=2)

        # [batch_size, classes, hidden_size]
        target_mol_encs = target_mol_encs.view(-1, 1, hidden_size)
        target_mol_encs = target_mol_encs.repeat(1, classes, 1)

        # [batch_size * classes]
        distances_flat = F.pairwise_distance(
            support_mol_encs.view(-1, hidden_size),
            target_mol_encs.view(-1, hidden_size),
            p=2)

        # [batch_size, classes]
        distances = distances_flat.view(-1, classes)

        return -distances


class RelationNet(PairwiseSimilarity):

    def __init__(self, hparams):
        super(RelationNet, self).__init__(hparams)
        self.bilinear = nn.Linear(hparams.enc_hidden_size, hparams.enc_hidden_size)

    def compute_similarity(self, support_mol_encs, target_mol_encs):
        batch_size = support_mol_encs.size(0)
        hidden_size = self.hparams.enc_hidden_size
        classes = self.hparams.num_classes

        # [batch_size, classes, hidden_size]
        support_mol_encs = support_mol_encs.mean(dim=2)
        support_mol_encs = self.bilinear(support_mol_encs)

        # [batch_size, classes, hidden_size]
        target_mol_encs = target_mol_encs.view(-1, 1, hidden_size)
        target_mol_encs = target_mol_encs.expand(batch_size, classes, hidden_size)

        # [batch_size, classes]
        scores = (support_mol_encs * target_mol_encs).sum(dim=-1)
        return scores


class SetTransformer(PairwiseSimilarity):

    def __init__(self, hparams):
        super(SetTransformer, self).__init__(hparams)
        self.support_emb = nn.Parameter(torch.randn(hparams.enc_hidden_size))
        self.target_emb = nn.Parameter(torch.randn(hparams.enc_hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparams.enc_hidden_size, nhead=hparams.num_enc_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams.num_enc_layers)

    def compute_similarity(self, support_mol_encs, target_mol_encs):
        hidden_size = self.hparams.enc_hidden_size
        k = self.hparams.k_shot
        classes = self.hparams.num_classes

        support_mol_encs += self.support_emb
        target_mol_encs += self.target_emb

        # [batch_size, classes, k + 1, hidden_size]
        target_mol_encs = target_mol_encs.view(-1, 1, hidden_size)
        target_mol_encs = target_mol_encs.repeat(1, classes, 1)
        target_mol_encs = target_mol_encs.view(-1, classes, 1, hidden_size)
        combined_mol_encs = torch.cat([support_mol_encs, target_mol_encs], dim=2)

        # [batch_size * classes, k + 1, hidden_size]
        combined_mol_encs_flat = combined_mol_encs.view(-1, k + 1, hidden_size)

        # [batch_size * classes, k + 1, hidden_size]
        combined_mol_encs_flat = self.transformer_encoder(combined_mol_encs_flat)

        # [batch_size, classes, k + 1, hidden_size]
        combined_mol_encs = combined_mol_encs.view(-1, classes, k + 1, hidden_size)

        # [batch_size, classes, hidden_size]
        support_mol_encs = combined_mol_encs[:, :, :k, :].mean(dim=2)

        # [batch_size, classes, hidden_size]
        target_mol_encs = combined_mol_encs[:, :, -1, :]

        # [batch_size * classes]
        distances_flat = F.pairwise_distance(
            support_mol_encs.view(-1, hidden_size),
            target_mol_encs.view(-1, hidden_size),
            p=2)

        # [batch_size, classes]
        distances = distances_flat.view(-1, classes)

        return -distances

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_enc_heads", type=int, default=4)
        parser.add_argument("--num_enc_layers", type=int, default=4)
        return parser


class MAML(chembl.ChEMBLFewShot):

    def __init__(self, hparams):
        super(MAML, self).__init__(hparams)
        self.classifier = nn.Linear(hparams.enc_hidden_size, hparams.num_classes)

    def forward(self, inputs, mode="train"):
        mol_graph, mol_features, targets = inputs

        # [classes * k + 1, hidden_size]
        mol_encs = self.encoder(mol_graph, mol_features)

        # [classes * k + 1, num_classes]
        scores = self.classifier(mol_encs)
        if mode == "train":
            scores = scores[:-1]
            targets = targets[:-1]
        else:
            scores = scores[-1:]
            targets = targets[-1:]

        # [batch_size]
        predictions = torch.argmax(scores, dim=1).detach()

        outputs = dict(preds=predictions)

        # Get loss & accuracy.
        if targets is not None:
            outputs["loss"] = F.cross_entropy(scores, targets, reduction="none")
            outputs["acc"] = predictions.eq(targets).float()

        return outputs

    def get_outputs(self, inputs):
        torch.set_grad_enabled(True)
        if self.training:
            fine_tune_steps = self.hparams.train_fine_tune_steps
        else:
            fine_tune_steps = self.hparams.test_fine_tune_steps

        inner_opt = optim.SGD(self.parameters(), lr=self.hparams.fine_tune_lr)

        with higher.innerloop_ctx(
            model=self,
            opt=inner_opt,
            copy_initial_weights=not self.training,
            track_higher_grads=self.training,
        ) as (fnet, diffopt):
            # Inner loop optimization.
            fnet.train()
            for _ in range(fine_tune_steps):
                train_outputs = fnet(inputs, mode="train")
                diffopt.step(train_outputs["loss"].mean())

            # Test time prediction.
            fnet.eval()
            test_outputs = fnet(inputs, mode="test")

        return test_outputs

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.add_argument("--fine_tune_lr", type=float, default=0.1)
        parser.add_argument("--train_fine_tune_steps", type=int, default=1)
        parser.add_argument("--test_fine_tune_steps", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--accumulate_grad_batches", type=int, default=20)
        return parser
