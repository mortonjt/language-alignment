#!/bin/bash
#
#SBATCH --job-name=attn_dist
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p ccb


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo `which python`
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission

#for epoch in 1 2 3 4 5
for epoch in 5 uniref90
do
    embeds=$DATADIR/results/embeddings/attn/epoch${epoch}
    triples=$DATADIR/data/pfam/subset_triples.txt
    triples=$DATADIR/data/pfam/pfam_benchmark_pairs_50k.txt
    out=$DATADIR/results/distances/elmo/elmo_epoch${epoch}_distances_full.txt
    python scripts/get_distances.py $embeds $triples $out
done
