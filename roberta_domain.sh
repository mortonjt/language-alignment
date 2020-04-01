#!/bin/bash
#
#SBATCH --job-name=attn_dom
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
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
pairs_dir=$DATADIR/data/alignment/domain_pairs_split

#for fname in $pairs_dir/*
#do
#     echo $fname
#     f=`basename $fname`
#     out=results/alignments
#     cmd="python scripts/get_alignment.py $embeds $fname $out $components"
#     sbatch -p ccb --ntasks 1 --cpus-per-task 1 --wrap "$cmd"
#     # python get_alignment.py $embeds $fname $out $components
#done
#for epoch in 1 2 3 4 5
for epoch in 5 uniref90
do
   embeds=$DATADIR/results/embeddings/attn/epoch${epoch}/
   triplets=$DATADIR/data/domain/subsampled_triplets_1k.tsv
   out=$DATADIR/results/domains/attn/attn_domain_epoch${epoch}.txt
   python scripts/domain_detection.py $embeds $triplets $out
done
