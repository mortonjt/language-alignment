#!/bin/bash
#
#SBATCH --job-name=elmo
#SBATCH --output=stdout_elmo.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=16
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:01

source ~/venvs/transformers-torch/bin/activate
module load slurm
module load cuda/10.0.130_410.48
module load cudnn/v7.6.2-cuda-10.0

DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
in_file=data/raw/combined.fasta
#in_file=$DATADIR/data/pfam/pfam_benchmark_seqs.fasta
epoch=5
in_file=../results/permute.fasta
model_path=$DATADIR/data/elmo/model_epoch_${epoch}.hdf5
results_dir=../results/permute_elmo_embeds
python scripts/extract_elmo.py $in_file $model_path $results_dir
echo "Epoch ${epoch} done."
