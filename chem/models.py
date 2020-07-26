"""Few-Shot learning for ChEMBL."""

import argparse
import torch
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


class FinetuneProtoNet(ProtoNet):

    def finetune(self, inputs):
        mol_graph, mol_features, targets = inputs

        # [classes * k, hidden_size]
        mol_encs = self.encoder(mol_graph, mol_features)[:-1]

        # [classes * k, hidden_size]
        hidden_size = self.hparams.enc_hidden_size
        k = self.hparams.k_shot
        classes = self.hparams.num_classes
        mol_encs = mol_encs.view(classes * k, hidden_size)

        # [classes, k, hidden_size]
        support_mol_encs = mol_encs.view(classes, k, hidden_size)

        # [classes, hidden_size]
        mean_mol_encs = support_mol_encs.mean(dim=1)

        # [classes * k, classes, hidden_size]
        support_mol_encs = support_mol_encs.unsqueeze(2).repeat(1, 1, classes, 1)
        support_mol_encs = support_mol_encs.view(classes * k, classes, hidden_size)

        # [classes * k, classes, hidden_size]
        mean_mol_encs = mean_mol_encs.unsqueeze(0).repeat(classes * k, 1, 1)

        # [classes * k * classes]
        distances_flat = F.pairwise_distance(
            support_mol_encs.view(-1, hidden_size),
            mean_mol_encs.view(-1, hidden_size),
            p=2)

        # [classes * k, classes]
        scores = -distances_flat.view(-1, classes)

        # [classes * k]
        targets = targets[:-1].view(classes * k)

        loss = F.cross_entropy(scores, targets)

        return loss

    def get_outputs(self, inputs):
        if self.training:
            return self.forward(inputs)

        inner_opt = optim.Adam(self.parameters(), lr=self.hparams.fine_tune_lr)

        with higher.innerloop_ctx(
            model=self,
            opt=inner_opt,
            copy_initial_weights=True,
            track_higher_grads=False,
        ) as (fnet, diffopt):
            # Inner loop optimization.
            initial = fnet.forward(inputs)
            initial = {k: v.detach() for k, v in initial.items()}

            torch.set_grad_enabled(True)
            fnet.train()
            for _ in range(self.hparams.fine_tune_steps):
                loss = fnet.finetune(inputs)
                diffopt.step(loss)

            import pdb; pdb.set_trace()
            # Test time prediction.
            torch.set_grad_enabled(False)
            fnet.eval()
            final = fnet.forward(inputs)
            final = {k: v.detach() for k, v in final.items()}

        return {"initial": initial, "final": final}

    def validation_epoch_end(self, outputs):
        initial_loss = torch.cat([x["initial"]["loss"] for x in outputs]).mean().item()
        final_loss = torch.cat([x["final"]["loss"] for x in outputs]).mean().item()
        initial_acc = torch.cat([x["initial"]["acc"] for x in outputs]).mean().item()
        final_acc = torch.cat([x["final"]["acc"] for x in outputs]).mean().item()
        tensorboard_logs = {"initial_val_loss": initial_loss,
                            "final_val_loss": final_loss,
                            "initial_val_acc": initial_acc,
                            "final_val_acc": final_acc}
        tqdm_dict = {"initial_val_acc": initial_acc,
                     "final_val_acc": final_acc}
        return {"val_loss": final_loss,
                "progress_bar": tqdm_dict,
                "log": tensorboard_logs}

    def test_epoch_end(self, outputs):
        initial_loss = torch.cat([x["initial"]["loss"] for x in outputs]).mean().item()
        final_loss = torch.cat([x["final"]["loss"] for x in outputs]).mean().item()
        initial_acc = torch.cat([x["initial"]["acc"] for x in outputs]).mean().item()
        final_acc = torch.cat([x["final"]["acc"] for x in outputs]).mean().item()
        tensorboard_logs = {"initial_test_loss": initial_loss,
                            "final_test_loss": final_loss,
                            "initial_test_acc": initial_acc,
                            "final_test_acc": final_acc}
        tqdm_dict = {"initial_test_acc": initial_acc,
                     "final_test_acc": final_acc}
        return {"test_loss": final_loss,
                "progress_bar": tqdm_dict,
                "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], conflict_handler="resolve", add_help=False)
        parser.add_argument("--fine_tune_lr", type=float, default=0.001)
        parser.add_argument("--fine_tune_steps", type=int, default=100)
        parser.add_argument("--test_batch_size", type=int, default=1)
        return parser
