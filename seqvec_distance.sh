#!/bin/bash
#
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
mkdir -p $DATADIR/results/pfam/seqvec/
embeds=$DATADIR/results/embeddings/seqvec/
triples=$DATADIR/data/pfam/pfam_benchmark_pairs_50k.txt
out=$DATADIR/results/pfam/seqvec/pfam.txt
python scripts/get_distances.py $embeds $triples $out True


# SCOP triples
mkdir -p $DATADIR/results/scop/seqvec/
embeds=$DATADIR/results/embeddings/seqvec/
triples=$DATADIR/data/scop/scop_triples.txt
out=$DATADIR/results/scop/seqvec/scop.txt
python scripts/get_distances.py $embeds $triples $out True
