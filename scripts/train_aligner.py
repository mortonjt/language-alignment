import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
import os
import re
from Bio import SeqIO

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import MeanAligner, SSAaligner, CCAaligner
from language_alignment.losses import TripletLoss
from language_alignment.dataset import AlignmentDataset
from language_alignment.dataset import collate_alignment_pairs, seq2onehot
from tape import TAPETokenizer


class LightningAligner(pl.LightningModule):

    def __init__(self, args):
        super(LightningAligner, self).__init__()
        cls, path = pretrained_language_models[args.arch]
        self.args = args
        if args.lm is not None:
            path = args.lm
        align_fun = self.aligner_type()
        self.model = AlignmentModel(aligner=align_fun)
        self.model.load_language_model(cls, path)

        seqs = list((SeqIO.parse(open(self.args.fasta), format='fasta')))
        self.seqs = {x.id: x.seq for x in seqs}
        self.cfxn = lambda x: collate_alignment_pairs(x)
        self.cfxn = collate_alignment_pairs
        self.triplet_loss = TripletLoss()

        if args.arch == 'bert':
            tr = TAPETokenizer(vocab='iupac')
            f = lambda x: torch.tensor([tr.encode(x)]).squeeze()
            self.tokenizer = f
        elif args.arch == 'unirep':
            tr = TAPETokenizer(vocab='unirep')
            f = lambda x: torch.tensor([tr.encode(x)]).squeeze()
            self.tokenizer = f
        else:
            self.tokenizer = seq2onehot

    def forward(self, x, y):
        return self.model.forward(x, y)

    def aligner_type(self):
        input_dim = self.args.lm_embed_dim
        embed_dim = self.args.aligner_embed_dim
        if self.args.aligner == 'cca':
            align_fun = CCAaligner(input_dim, embed_dim)
        elif self.args.aligner == 'ssa':
            align_fun = SSAaligner(input_dim, embed_dim)
        else:
            align_fun = MeanAligner()
        return align_fun

    def initialize_logging(self, root_dir='./', logging_path=None):
        if logging_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            logging_path = "_".join([basename, suffix])
        full_path = root_dir + logging_path
        writer = SummaryWriter(full_path)
        return writer

    def train_dataloader(self):
        train_pairs = pd.read_table(
            self.args.train_pairs, header=None, sep='\s+')
        train_dataset = AlignmentDataset(
            train_pairs, self.seqs, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, self.args.batch_size,
                                      shuffle=True, collate_fn=self.cfxn,
                                      num_workers=self.args.num_workers)
        return train_dataloader

    def valid_dataloader(self):
        valid_pairs = pd.read_table(
            self.args.valid_pairs, header=None, sep='\s+')
        valid_dataset = AlignmentDataset(
            valid_pairs, self.seqs, self.tokenizer)
        valid_dataloader = DataLoader(valid_dataset, self.args.batch_size,
                                      shuffle=True, collate_fn=self.cfxn,
                                      num_workers=self.args.num_workers)
        return valid_dataloader

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        self.model.train()
        xy = self.model(x, y)
        xz = self.model(x, z)
        loss = self.triplet_loss(xy, xz)
        return {'loss': loss}

    def valid_step(self, batch, batch_idx):
        x, y, z = batch
        self.model.eval()
        xy = self.model(x, y)
        xz = self.model(x, z)
        loss = self.triplet_loss(xy, xz)
        return {'val_loss': loss.item()}

    def configure_optimizers(self):
        if not self.args.finetune:
            for p in self.model.lm.parameters():
                p.requires_grad = False
            grad_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            optimizer = torch.optim.RMSprop(
                grad_params, lr=self.args.learning_rate, weight_decay=self.args.reg_par)

        else:
            optimizer = optim.SGD(
                [
                    {'params': self.model.lm.parameters(), 'lr': 5e-6},
                    {'params': self.model.aligner_fun.parameters(),
                     'lr': self.args.learning_rate}
                ]
            )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--train-pairs', help='Training pairs file', required=True)
        parser.add_argument('--valid-pairs', help='Validation pairs file', required=True)
        parser.add_argument('--fasta', help='Fasta file', required=True)
        parser.add_argument('-m','--lm', help='Path to pretrained model',
                            required=False, default=None)
        parser.add_argument('-c','--arch',
                            help=('Pretrained model type (choices include onehot, '
                                  'elmo, bert, unirep and roberta'),
                            required=False, default='bert')
        parser.add_argument('-a','--aligner',
                            help='Aligner type. Choices include (mean, cca, ssa).',
                            required=False, type=str, default='mean')
        parser.add_argument('--lm-embed-dim', help='Language model embedding dimension.',
                            required=False, type=int)
        parser.add_argument('--aligner-embed-dim', help='Aligner embedding dimension.',
                            required=False, type=int)
        parser.add_argument('--reg-par', help='Regularization.',
                            required=False, type=float, default=1e-5)
        parser.add_argument('--max-len', help='Maximum length of protein', default=1024,
                            required=False, type=str)
        parser.add_argument('--learning-rate', help='Learning rate',
                            required=False, type=float, default=5e-5)
        parser.add_argument('--batch-size', help='Training batch size',
                            required=False, type=int, default=32)
        parser.add_argument('--finetune', help='Perform finetuning (does not work with mean)',
                            default=False, required=False, type=bool)
        parser.add_argument('--epochs', help='Training batch size',
                            required=False, type=int, default=10)
        parser.add_argument('-o','--output-directory', help='Output directory of model results',
                            required=True)
        return parser


def main(args):
    model = LightningAligner(args)
    trainer = Trainer(
        max_nb_epochs=args.epochs,
        gpus=args.gpus,
        nb_gpu_nodes=args.nodes,
        accumulate_grad_batches=args.grad_accum
    )

    ckpt_path = os.path.join(
        trainer.default_save_path,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )
    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        verbose=True,
        period=1,
    )
    trainer.checkpoint_callback = checkpoint_callback

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    parser = LightningAligner.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
