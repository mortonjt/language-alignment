#!/bin/bash
#
#SBATCH --job-name=sw
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=4
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p genx

source ~/.bashrc
source ~/venvs/transformers-torch/bin/activate
conda activate alignment
DATADIR=/mnt/home/jmorton/research/gert/icml2020/language-alignment
# parse results
echo 'pfam'
outdir=$DATADIR/results/distances/sw/pfam
dataset=pfam
mkdir -p $outdir
test_file=$DATADIR/data/$dataset/data-bin/test.txt
python scripts/sw_align.py \
    -f $DATADIR/data/alignment-train/testing-set/test_pfam.fa \
    -p $test_file \
    -o $outdir/pfam_results.txt

echo 'scop'
outdir=$DATADIR/results/distances/sw/scop
dataset=scop
mkdir -p $outdir
test_file=$DATADIR/data/$dataset/data-bin/test.txt
python scripts/sw_align.py \
    -f $DATADIR/data/alignment-train/testing-set/test_scop.fa \
    -p $test_file \
    -o $outdir/scop_results.txt
