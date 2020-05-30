#!/bin/bash
#SBATCH -p genx
#SBATCH -c 18
source ~/.bashrc
conda activate alignment

DATADIR=/mnt/home/jmorton/research/gert/icml2020/language-alignment

inpath=$DATADIR/data/alignment-train/testing-set/test_pfam.fa
outdir=$DATADIR/results/distances/hmmer/pfam
mkdir -p $outdir
phmmer -o $outdir/pfam.txt --cpu 18 $inpath $inpath --max

inpath=$DATADIR/data/alignment-train/testing-set/test_scop.fa
outdir=$DATADIR/results/distances/hmmer/scop
mkdir -p $outdir
phmmer -o $outdir/scop.txt --cpu 18 $inpath $inpath --max

# parse results
echo 'pfam'
dataset=pfam
outdir=$DATADIR/results/distances/hmmer/pfam
hmmer_results=$DATADIR/results/distances/hmmer/pfam/pfam.txt
hmmer_alignments=$DATADIR/results/distances/hmmer/pfam/pfam_alignments.txt
test_file=$DATADIR/data/$dataset/data-bin/test.txt
python scripts/parse_alignment.py \
    --path $hmmer_results \
    --aligner hmmer \
    --out-path $hmmer_alignments
python scripts/evaluate_blast.py \
    -p $test_file \
    -a $hmmer_alignments \
    -o $outdir/pfam_results.txt

echo 'scop'
dataset=scop
outdir=$DATADIR/results/distances/hmmer/scop
hmmer_results=$DATADIR/results/distances/hmmer/scop/scop.txt
hmmer_alignments=$DATADIR/results/distances/hmmer/scop/scop_alignments.txt
test_file=$DATADIR/data/$dataset/data-bin/test.txt
python scripts/parse_alignment.py \
    --path $hmmer_results \
    --aligner hmmer \
    --out-path $hmmer_alignments
python scripts/evaluate_blast.py \
    -p $test_file \
    -a $hmmer_alignments \
    -o $outdir/scop_results.txt
