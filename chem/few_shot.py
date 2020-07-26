"""Few-Shot learning for ChEMBL."""

import argparse
import pytorch_lightning as pl
import models


def main(args):
    pl.seed_everything(args.seed)
    model = models.SetTransformer(hparams=args)
    print(model)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = models.SetTransformer.add_task_specific_args(parser)
    parser = models.SetTransformer.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
