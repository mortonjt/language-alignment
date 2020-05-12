import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
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
from torch.nn.utils import clip_grad_norm_

from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import MeanAligner, SSAaligner, CCAaligner
from language_alignment.losses import TripletLoss
from language_alignment.dataset import AlignmentDataset
from language_alignment.dataset import collate_alignment_pairs


def aligner_type(args):
    input_dim = args.lm_embed_dim
    embed_dim = args.aligner_embed_dim
    max_len = args.max_len
    device = 'cuda' if args.gpu else 'cpu'
    if args.aligner == 'cca':
        align_fun = CCAaligner(input_dim, embed_dim, device=device)
    elif args.aligner == 'ssa':
        align_fun = SSAaligner(input_dim, embed_dim)
    else:
        align_fun = MeanAligner()
    return align_fun

def initialize_logging(root_dir='./', logging_path=None):
    if logging_path is None:
        basename = "logdir"
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        logging_path = "_".join([basename, suffix])
    full_path = root_dir + logging_path
    writer = SummaryWriter(full_path)
    return writer

def init_model(args):
    cls, path = pretrained_language_models[args.arch]
    device = 'cuda' if args.gpu else 'cpu'
    if args.lm is not None:
        path = args.lm
    align_fun = aligner_type(args)
    model = AlignmentModel(aligner=align_fun)
    model.load_language_model(cls, path, device=device)
    model.to(device)
    return model, device

def init_dataloaders(args, device):
    seqs = list((SeqIO.parse(open(args.fasta), format='fasta')))
    seqs = {x.id: x.seq for x in seqs}
    cfxn = lambda x: collate_alignment_pairs(x, device)
    train_pairs = pd.read_table(args.train_pairs, header=None, sep='\s+')
    valid_pairs = pd.read_table(args.valid_pairs, header=None, sep='\s+')
    train_dataset = AlignmentDataset(train_pairs, seqs)
    valid_dataset = AlignmentDataset(valid_pairs, seqs)
    train_dataloader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, collate_fn=cfxn)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size,
                                  shuffle=True, collate_fn=cfxn)
    return train_dataloader, valid_dataloader

def make_train_step(model, triplet_loss, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y, z):
        model.train()
        optimizer.zero_grad()
        xy = model(x, y)
        xz = model(x, z)
        loss = triplet_loss(xy, xz)
        loss.backward()
        grad_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer.step()
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step

def make_valid_step(model, triplet_loss, scheduler):
    # Builds function that performs a step in the valid loop
    def valid_step(x, y, z):
        model.eval()
        xy = model(x, y)
        xz = model(x, z)
        loss = triplet_loss(xy, xz)
        return loss.item()
        scheduler.step()
    # Returns the function that will be called inside the valid loop
    return valid_step

def checkpoint_step(args, model, avg_valid_loss, best_valid_loss):
    torch.save(model.state_dict(), args.output_directory + '/model_current.pt')
    if avg_valid_loss < best_valid_loss:
        torch.save(model.state_dict(), args.output_directory + '/model_best.pt')
        best_valid_loss = avg_valid_loss

def main(args):
    # set seed for debugging
    # torch.manual_seed(0)
    # Initialize model
    model, device = init_model(args)
    # Initialize tensorboard
    writer = initialize_logging(root_dir=args.output_directory,
                                logging_path='/logs/')
    # Setup Dataloader
    train_dataloader, valid_dataloader = init_dataloaders(args, device)
    # optimizer
    if not args.finetune:
        for p in model.lm.parameters():
            p.requires_grad = False
        grad_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.RMSprop(
            grad_params, lr=args.learning_rate, weight_decay=args.reg_par)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    else:
        optimizer = optim.SGD(
            [
                {'params': model.lm.parameters(), 'lr': 1e-6},
                {'params': model.aligner_fun.parameters(), 'lr': args.learning_rate}
            ]
        )
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    # Define loss function
    triplet_loss = TripletLoss()
    # Creates the train_step function
    train_step = make_train_step(model, triplet_loss, optimizer)
    # Creates the valid_step function
    valid_step = make_valid_step(model, triplet_loss, scheduler)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, valid_loss, best_valid_loss = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            loss = train_step(*batch)
            if loss != loss:
                raise ValueError("Loss is nan")
            train_loss += loss
            if batch_idx % 100 == 0:
                print("Batch {}/{}.  Batch loss: {}.".format(
                    batch_idx, len(train_dataloader), train_loss / (batch_idx+1)))

        for batch_idx, batch in enumerate(valid_dataloader):
            valid_loss += valid_step(*batch)

        avg_train_loss = train_loss / len(train_dataloader)
        avg_valid_loss = valid_loss / len(valid_dataloader)
        writer.add_scalar('training_loss', avg_train_loss, epoch)
        writer.add_scalar('validation_loss', avg_valid_loss, epoch)
        best_valid_loss = checkpoint_step(args, model,
                                          avg_valid_loss, best_valid_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train-pairs', help='Training pairs file', required=True)
    parser.add_argument('--valid-pairs', help='Validation pairs file', required=True)
    parser.add_argument('--fasta', help='Fasta file', required=True)
    parser.add_argument('-m','--lm', help='Path to pretrained model',
                        required=False, default=None)
    parser.add_argument('-c','--arch',
                        help='Pretrained model type (choices include onehot, elmo and roberta',
                        required=False, default='elmo')
    parser.add_argument('-a','--aligner',
                        help='Aligner type. Choices include (mean, cca, ssa).',
                        required=False, type=str, default='mean')
    parser.add_argument('--lm-embed-dim', help='Language model embedding dimension.',
                        required=False, type=int, default=1024)
    parser.add_argument('--aligner-embed-dim', help='Aligner embedding dimension.',
                        required=False, type=int, default=128)
    parser.add_argument('--max-len', help='Maximum length of protein', default=1024,
                        required=False, type=str)
    parser.add_argument('--learning-rate', help='Learning rate',
                        required=False, type=float, default=1e-3)
    parser.add_argument('--reg-par', help='Regularization.',
                        required=False, type=float, default=1e-5)
    parser.add_argument('--batch-size', help='Training batch size (needs to be 1 for cca)',
                        required=False, type=int, default=32)
    parser.add_argument('--epochs', help='Training batch size',
                        required=False, type=int, default=10)
p    parser.add_argument('--finetune', help='Perform finetuning (does not work with mean)',
                        default=False, required=False, type=bool)
    parser.add_argument('-g','--gpu', help='Use GPU or not', default=False,
                        required=False, type=bool)
    parser.add_argument('-o','--output-directory', help='Output directory of model results',
                        required=True)
    args = parser.parse_args()
    main(args)
