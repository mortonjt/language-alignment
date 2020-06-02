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


# PFAM triples
mkdir -p $DATADIR/results/pfam/transformer/
embeds=$DATADIR/results/embeddings/transformer
triples=$DATADIR/data/pfam/pfam_benchmark_pairs_50k.txt
out=$DATADIR/results/pfam/transformer/pfam.txt
python scripts/get_distances.py $embeds $triples $out

# SCOP triples
mkdir -p $DATADIR/results/scop/transformer/
embeds=$DATADIR/results/embeddings/transformer
triples=$DATADIR/data/scop/scop_triples.txt
out=$DATADIR/results/scop/transformer/pfam.txt
python scripts/get_distances.py $embeds $triples $out
