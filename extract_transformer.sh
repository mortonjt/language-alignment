#!/bin/bash
#
#SBATCH --job-name=attn
#SBATCH --output=stdout_attn.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=4
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:01

source ~/venvs/transformers-torch/bin/activate
module load slurm
module load cuda/10.0.130_410.48
module load cudnn/v7.6.2-cuda-10.0

# NOTE may want to change these paths
DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
in_file=$DATADIR/data/raw/combined.fasta
#in_file=$DATADIR/data/pfam/pfam_benchmark_seqs.fasta

#in_file=../results/permute.fasta
#in_file=$DATADIR/data/raw/permuted.fasta
#model_path=$DATADIR/data/attn/checkpoint_uniref90.pt
model_path=/mnt/home/jmorton/research/gert/roberta-checkpoints/pfam-transformer/checkpoint_best.pt
#results_dir=$DATADIR/results/embeddings/attn/epoch${epoch}/
results_dir=$DATADIR/results/embeddings/transformer/
mkdir -p $results_dir
python scripts/extract_transformer.py $in_file $model_path $results_dir


# epoch=5
# in_file=../results/permute.fasta
# model_path=$DATADIR/data/attn/checkpoint_uniref90.pt
# results_dir=$DATADIR/results/embeddings/attn/epoch${epoch}/
# mkdir -p $results_dir
# python scripts/extract_attention.py $in_file $model_path $results_dir
# echo "Epoch ${epoch} done."
