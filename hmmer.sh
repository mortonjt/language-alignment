#!/bin/bash
#SBATCH -p genx
#SBATCH -c 18
source ~/.bashrc
conda activate alignment

DATADIR=/mnt/home/jmorton/research/gert/icml2020/language-alignment

inpath=$DATADIR/data/alignment-train/testing-set/test_pfam.fa
outdir=$DATADIR/results/distances/hmmer/pfam
mkdir -p $outdir
phmmer -o $outdir/pfam.txt --cpu 18 $inpath $inpath

inpath=$DATADIR/data/alignment-train/testing-set/test_scop.fa
outdir=$DATADIR/results/distances/hmmer/scop
mkdir -p $outdir
phmmer -o $outdir/scop.txt --cpu 18 $inpath $inpath
