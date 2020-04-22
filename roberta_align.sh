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
components=80
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission

## domain alignment benchmark
# pairs_dir=$DATADIR/data/alignment/domain_pairs_split
# epoch=uniref90
# embeds=$DATADIR/results/embeddings/attn/epoch${epoch}/
# for fname in $pairs_dir/*
# do
#      echo $fname
#      f=`basename $fname`
#      out=$DATADIR/results/alignments/attn_epoch${epoch}
#      mkdir -p $out
#      cmd="python scripts/get_alignment.py $embeds $fname $out $components"
#      sbatch -p genx --ntasks 1 --cpus-per-task 1 --wrap "$cmd"
#      # python get_alignment.py $embeds $fname $out $components
# done

## structure alignment
pairs_dir=
epoch=uniref90
embeds=/mnt/ceph/users/jmorton/embeddings/roberta/malisam/
fname=$DATADIR/data/alignment/malisam/analog_ids.txt

echo $fname
f=`basename $fname`
out=$DATADIR/results/alignments/malisam/roberta_${epoch}
mkdir -p $out
cmd="python scripts/get_alignment.py $embeds $fname $out $components"
sbatch -p genx --ntasks 1 --cpus-per-task 1 --wrap "$cmd"
#python scripts/get_alignment.py $embeds $fname $out $components

embeds=/mnt/ceph/users/jmorton/embeddings/roberta/malidup/
out=$DATADIR/results/alignments/malidup/roberta_${epoch}
mkdir -p $out
fname=$DATADIR/data/alignment/malidup/dup_ids.txt
cmd="python scripts/get_alignment.py $embeds $fname $out $components"
sbatch -p genx --ntasks 1 --cpus-per-task 1 --wrap "$cmd"
