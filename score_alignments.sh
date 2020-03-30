#!/bin/bash
#
#SBATCH --job-name=score_align
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH --ntasks 12
#SBATCH -p ccb


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo `which python`


DATADIR=/mnt/ceph/users/jmorton/icml-final-submission

epoch=uniref90
pair_file=$DATADIR/data/alignment/domain_pairs.txt
fasta_file=$DATADIR/data/raw/combined.fasta
alignments=$DATADIR/results/alignments/attn_epoch${epoch}
dom_file=$DATADIR/data/raw/swissprot-pfam-domains.csv
out_file=$DATADIR/results/alignments/attn_${epoch}_${pair}_alignment_scores.txt
# python scripts/score_alignment.py $alignments $fasta_file $pair_file $dom_file $out_file
for pair_file in $DATADIR/data/alignment/domain_pairs_split/*
do
    pair=`basename $pair_file`
    out_file=$DATADIR/results/alignments/attn_${epoch}_${pair}_alignment_scores.txt
    echo $alignments
    echo $fasta_file
    echo $pair_file
    echo $out_file

    cmd="python scripts/score_alignment.py $alignments $fasta_file $pair_file $dom_file $out_file"
    echo $cmd
    sbatch -p ccb --wrap "$cmd"
done
#echo "Attention $epoch is done"

epoch=5

alignments=$DATADIR/results/alignments/elmo
pair_file=$DATADIR/data/alignment/domain_pairs.txt
dom_file=$DATADIR/data/raw/swissprot-pfam-domains.csv
out_file=$DATADIR/results/alignments/elmo_alignment_scores.txt
fasta_file=$DATADIR/data/raw/combined.fasta
out_file=$DATADIR/results/alignments/elmo_${epoch}_${pair}_alignment_scores.txt
# python scripts/score_alignment.py $alignments $fasta_file $pair_file $dom_file $out_file
for pair_file in $DATADIR/data/alignment/domain_pairs_split/*
do
    echo $alignments
    echo $fasta_file
    echo $pair_file
    echo $out_file
    pair=`basename $pair_file`

    out_file=$DATADIR/results/alignments/elmo_${epoch}_${pair}_alignment_scores.txt
    #python scripts/score_alignment.py $alignments $fasta_file $pair_file $dom_file $out_file
    cmd="python scripts/score_alignment.py $alignments $fasta_file $pair_file $dom_file $out_file"
    echo $cmd
    sbatch -p ccb --wrap "$cmd"

done
#echo "Elmo $epoch is done"
