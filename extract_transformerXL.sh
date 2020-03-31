#!/bin/bash
#
#SBATCH --job-name=bertXL
#SBATCH --output=stdout_transformerXL.txt
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
model_dir=/mnt/home/aelnaggar/ceph/Jamie/TransformerXL

# train=Uniref100
# model_path=$model_dir/${train}/model.pt
# results_dir=$DATADIR/results/embeddings/transformerXL/${train}/
# mkdir -p $results_dir
# python scripts/extract_transformerXL.py $in_file $model_path $results_dir
# echo "${train} done."

train=BFD100
model_path=$model_dir/${train}/model-4800.pt
results_dir=$DATADIR/results/embeddings/transformerXL/${train}/
mkdir -p $results_dir
python scripts/extract_transformerXL.py $in_file $model_path $results_dir
echo "${train} done."
