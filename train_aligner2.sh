#!/bin/bash
#
#SBATCH --ntasks=4
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:01
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org


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
results_dir=results/aligner/model
model=/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert
#model=/mnt/home/jmorton/ceph/checkpoints/uniref90/base/
lm=roberta
method=cca
echo $method
echo $lm
python scripts/train_aligner.py \
    --train-pairs $train_file \
    --valid-pairs $valid_file \
    --fasta $fasta_file \
    --arch $lm \
    --batch-size 1 \
    --aligner $method \
    --learning-rate 1e-3 \
    --reg-par 1 \
    --epochs 5 \
    -m $model \
    --max-len 1024 \
    --lm-embed-dim 1024 \
    --aligner-embed-dim 1024 \
    --gpu True \
    -o results/aligner/${method}_${lm}_finetune_model_500k
