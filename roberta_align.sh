#!/bin/bash
#
#SBATCH --job-name=attn_align
#SBATCH --output=stdout_attn.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p ccb


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo 'running attention'
echo `which python`
components=40
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
pairs_dir=$DATADIR/data/alignment/domain_pairs_split

epoch=uniref90
embeds=$DATADIR/results/embeddings/attn/epoch${epoch}/

for fname in $pairs_dir/*
do
     echo $fname
     f=`basename $fname`
     out=$DATADIR/results/alignments/attn_epoch${epoch}
     mkdir -p $out
     cmd="python scripts/get_alignment.py $embeds $fname $out $components"
     sbatch -p ccb --ntasks 1 --cpus-per-task 1 --wrap "$cmd"
     # python get_alignment.py $embeds $fname $out $components
done
