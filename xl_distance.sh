#!/bin/bash
#
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p gen


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo `which python`
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission

# PFAM triples
# mkdir -p $DATADIR/results/pfam/transformerXL/
# for train in BFD100 Uniref100
# do
#    embeds=$DATADIR/results/embeddings/transformerXL/${train}/
#    triples=$DATADIR/data/pfam/pfam_benchmark_pairs_50k.txt
#    out=$DATADIR/results/pfam/transformerXL/${train}_pfam.txt
#    python scripts/get_distances.py $embeds $triples $out
# done

# SCOP triples
mkdir -p $DATADIR/results/scop/transformerXL/
for train in BFD100 Uniref100
do
   embeds=$DATADIR/results/embeddings/transformerXL/${train}/
   triples=$DATADIR/data/scop/scop_triples.txt
   out=$DATADIR/results/scop/transformerXL/${train}_scop.txt
   python scripts/get_distances.py -i $embeds -t $triples -o $out -r True
done
