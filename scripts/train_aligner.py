import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import glob
import os
import re
from Bio import SeqIO

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from language_alignment import pretrained_language_models
from language_alignment.models import AlignmentModel
from language_alignment.layers import MeanAligner, SSAaligner, CCAaligner
from language_alignment.losses import TripletLoss
from language_alignment.dataset import NegativeSampler, AlignmentDataset,
from language_alignment.dataset import collate_alignment_pairs


def aligner_type(args):
    input_dim = args.lm_embed_dim
    embed_dim = args.aligner_embed_dim
    max_len = args.max_len
    if args.aligner == 'cca':
        align_fun = CCAaligner(input_dim, embed_dim, max_len)
    elif args.aligner == 'ssa':
        align_fun = SSAaligner(input_dim, embed_dim, max_len)
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
    cls, path = pretrained_language_models['onehot']
    device = 'cuda' if args.gpu else 'cpu'
    if args.lm is not None:
        path = args.lm
    align_fun = aligner_type(args)
    model = AlignmentModel(align_fun)
    model.load_language_model(cls, path, device=device)
    return model, device

def init_dataloaders(args, device):
    seqs = list((SeqIO.parse(open(arg.fasta))))
    sampler = NegativeSampler(seqs)
    cfxn = lambda x: collate_alignment_pairs(x, device)
    train_dataset = AlignmentDataset(args.train_pairs, sampler)
    valid_dataset = AlignmentDataset(args.valid_pairs, sampler)
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
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step

def make_valid_step(model, triplet_loss):
    # Builds function that performs a step in the valid loop
    def valid_step(x, y, z):
        model.eval()
        xy = model(x, y)
        xz = model(x, z)
        loss = triplet_loss(xy, xz)
        return loss.item()

    # Returns the function that will be called inside the valid loop
    return valid_step

def checkpoint_step(args, model, avg_valid_loss, best_valid_loss):
    torch.save(model.state_dict(), args.dir + '/model_current.pt')
    if avg_valid_loss < best_valid_loss:
        writer.add_text("Log", "Best validation loss achieved at %d." % n)
        torch.save(model.state_dict(), args.dir + '/model_best.pt')
        best_valid_loss = avg_valid_loss

def main(args):
    # Initialize model
    model, device = init_model(args)
    # Initialize tensorboard
    writer = initialize_logging(root_dir=args.output_directory,
                                logging_path='/logs/'):
    # Setup Dataloader
    train_dataloader, valid_dataloader = init_dataloaders(args, device)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # Define loss function
    triplet_loss = TripletLoss()
    # Creates the train_step function
    train_step = make_train_step(model, triplet_loss, optimizer)
    # Creates the valid_step function
    valid_step = make_valid_step(model, triplet_loss)
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, valid_loss, best_valid_loss = 0.0, 0.0, 0.0
        for batch_idx, batch in enumerate(train_dataloader):
            loss = train_step(*batch)
            train_loss += loss
            if batch_ix % 10 == 0:
                print("Batch {}/{}.  Batch loss: {}".format(
                    i, len(train_dataloader), loss))
        for batch_idx, batch in enumerate(valid_dataloader):
            valid_loss += train_step(*batch)

        avg_train_loss = train_loss / len(train_dataset)
        avg_valid_loss = valid_loss / len(valid_dataset)
        writer.add_scalar('training_loss', avg_train_loss)
        writer.add_scalar('validation_loss', avg_valid_loss)
        best_valid_loss = checkpoint_step(args, model,
                                          avg_valid_loss, best_valid_loss):


def __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--train-pairs', help='Training pairs file', required=True)
    parser.add_argument('--valid-pairs', help='Validation pairs file', required=True)
    parser.add_argument('--fasta', help='Fasta file', required=True)
    parser.add_argument('-m','--lm', help='Path to pretrained model',
                        required=False, default=None)
    parser.add_argument('-c','--arch',
                        help='Pretrained model type (choices include onehot, elmo and roberta',
                        required=False, default='elmo')
    parser.add_argument('-t','--finetune', help='Enable fine-tuning',
                        required=False, default=False)
    parser.add_argument('-a','--aligner', help='Aligner type. Choices include (mean, cca, ssa).',
                        required=False, type='str', default='mean')
    parser.add_argument('--lm-embed-dim', help='Language model embedding dimension.',
                        required=False, type='str')
    parser.add_argument('--aligner-embed-dim', help='Aligner embedding dimension.',
                        required=False, type='str')
    parser.add_argument('--max-len', help='Maximum length of protein', default=1024,
                        required=False, type='str')
    parser.add_argument('--learning-rate', help='Learning rate',
                        required=False, type=float, default=1e-3)
    parser.add_argument('--batch-size', help='Training batch size',
                        required=False, type=int, default=32)
    parser.add_argument('--epochs', help='Training batch size',
                        required=False, type=int, 10)
    parser.add_argument('-g','--gpu', help='Use GPU or not', default=False,
                        required=False, type='bool')
    parser.add_argument('-o','--output-directory', help='Output directory of model results',
                        required=True)
    args = parser.parse_args()
    main(args)
