"""Few-Shot learning for ChEMBL."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        predictions = torch.argmax(scores, dim=1)

        outputs = dict(scores=scores, preds=predictions)

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
        self.cls_emb = nn.Parameter(torch.randn(hparams.enc_hidden_size))
        self.support_emb = nn.Parameter(torch.randn(hparams.enc_hidden_size))
        self.target_emb = nn.Parameter(torch.randn(hparams.enc_hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hparams.enc_hidden_size, nhead=hparams.num_enc_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams.num_enc_layers)
        self.cls2score = nn.Linear(hparams.enc_hidden_size, 1)

    def run_transformer(self, input_embs):
        batch_size, seq_len, hidden_size = input_embs.size()
        cls_emb = self.cls_emb.view(1, 1, -1)
        cls_emb = cls_emb.expand(batch_size, seq_len, hidden_size)
        input_embs = torch.cat([cls_emb, input_embs], dim=1)
        hiddens = self.transformer_encoder(input_embs)
        cls_hidden = hiddens[:, 0, :]
        cls_score = self.cls2score(cls_hidden)
        return cls_score

    def compute_similarity(self, support_mol_encs, target_mol_encs):
        hidden_size = self.hparams.enc_hidden_size
        k = self.hparams.k_shot
        classes = self.hparams.num_classes

        # [batch_size, classes, k, hidden_size]
        support_mol_encs += self.support_emb

        # [batch_size, hidden_size]
        target_mol_encs += self.target_emb

        # [batch_size, classes, k + 1, hidden_size]
        target_mol_encs = target_mol_encs.view(-1, 1, hidden_size)
        target_mol_encs = target_mol_encs.repeat(1, classes, 1)
        target_mol_encs = target_mol_encs.view(-1, classes, 1, hidden_size)
        combined_mol_encs = torch.cat([support_mol_encs, target_mol_encs], dim=2)

        # [batch_size * classes, k + 1, hidden_size]
        combined_mol_encs_flat = combined_mol_encs.view(-1, k + 1, hidden_size)

        # [batch_size * classes]
        cls_scores = self.run_transformer(combined_mol_encs_flat)

        # [batch_size, classes]
        cls_scores = cls_scores.view(-1, classes)

        return cls_scores

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_enc_heads", type=int, default=8)
        parser.add_argument("--num_enc_layers", type=int, default=6)
        return parser


class MAML(chembl.ChEMBLFewShot):

    def __init__(self, hparams):
        super(MAML, self).__init__(hparams)
        self.classifier = nn.Linear(hparams.enc_hidden_size, hparams.num_classes)

    def forward(self, inputs):
        pass

    def get_task_input(self, inputs, i):
        pass

    def get_outputs(self, inputs):
        # task_num = self.hparams.batch_size
        # k = self.hparams.k_shot
        # classes = self.hparams.num_classes

        # inner_opt = torch.optim.SGD(net.parameters(), lr=1e-1)

        # for i in range(task_num):
        #     task_inputs = self.get_task_input(inputs, i)
        #     with higher.innerloop_ctx(
        #         model=self,
        #         opt=inner_opt,
        #         copy_initial_weights=not self.training,
        #         track_higher_grads=self.training,
        #     ) as (fnet, diffopt):
        #         for _ in range(n_inner_iter):
        #             spt_logits = fnet(x_spt[i])
        #             spt_loss = F.cross_entropy(spt_logits, y_spt[i])
        pass
