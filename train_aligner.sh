#!/bin/bash
#
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --exclusive


module load slurm
module load cuda/10.0.130_410.48
module load cudnn/v7.6.2-cuda-10.0

source ~/venvs/transformers-torch/bin/activate
cd /mnt/home/jmorton/research/gert/icml2020/language-alignment
datadir=/mnt/home/jmorton/research/gert/icml2020/language-alignment/data/alignment-train

train_file=$datadir/data-bin/train.txt
#train_file=$datadir/test-train.txt
valid_file=$datadir/data-bin/valid.txt
#valid_file=$datadir/test-valid.txt
fasta_file=$datadir/seqs.fasta
results_dir=results/aligner/model_4gpu
model=/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert
python scripts/train_aligner.py \
    --train-pairs $train_file \
    --valid-pairs $valid_file \
    --fasta $fasta_file \
    --arch 'roberta' \
    --batch-size 4 \
    --aligner ssa \
    --epochs 5 \
    -m $model \
    --gpus 4 \
    --grad-accum 32 \
    -o results/aligner/ssa_model
