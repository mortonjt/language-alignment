#!/bin/bash
#
#SBATCH --job-name=attn_scop
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p genx


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo `which python`
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
#mkdir -p results/scop/attn
#for epoch in 1 2 3 4 5
for epoch in 5 uniref90
do
    embeds=$DATADIR/results/embeddings/attn/epoch${epoch}
    triples=$DATADIR/data/scop/scop_triples.txt
    out=$DATADIR/results/scop/attn/attn_epoch${epoch}_distances.txt
    python scripts/get_distances.py $embeds $triples $out
    echo "Epoch $epoch is done"
done
