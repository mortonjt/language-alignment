#!/bin/bash
#
#SBATCH --job-name=bert_dom
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p ccb


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo 'running bert'
echo `which python`
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
pairs_dir=$DATADIR/data/alignment/domain_pairs_split


mkdir -p $DATADIR/results/domains/bert/
for train in BFD100 Uniref100
do
   embeds=$DATADIR/results/embeddings/bert/${train}/
   triplets=$DATADIR/data/domain/subsampled_triplets_1k.tsv
   out=$DATADIR/results/domains/bert/${train}_domain.txt
   python scripts/domain_detection.py $embeds $triplets $out
done
