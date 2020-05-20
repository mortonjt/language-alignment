#!/bin/bash
#
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclusive

module load slurm
# module load cuda/10.0.130_410.48
# module load cudnn/v7.6.2-cuda-10.0
module load cuda/10.1.243_418.87.00
module load cudnn/v7.6.2-cuda-10.1

source ~/venvs/transformers-torch/bin/activate
cd /mnt/home/jmorton/research/gert/icml2020/language-alignment
datadir=/mnt/home/jmorton/research/gert/icml2020/language-alignment/data/alignment-train

train_file=$datadir/data-bin/train.txt
#train_file=$datadir/test-train.txt
valid_file=$datadir/data-bin/valid.txt
#valid_file=$datadir/test-valid.txt
fasta_file=$datadir/seqs.fasta
results_dir=results/aligner/model
#model=/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert
#model=/mnt/home/jmorton/ceph/checkpoints/uniref90/base/
#seqvec
model=/mnt/home/jmorton/research/gert/icml2020/language-alignment/model
lm=seqvec
method=ssa
echo $method
echo $lm
dim=3072
reg_par=1
python scripts/train_aligner.py \
    --train-pairs $train_file \
    --valid-pairs $valid_file \
    --fasta $fasta_file \
    --arch $lm \
    --batch-size 1 \
    --aligner $method \
    --learning-rate 1e-3 \
    --reg-par $reg_par \
    --epochs 5 \
    -m $model \
    --max-len 768 \
    --lm-embed-dim $dim \
    --aligner-embed-dim 1024 \
    --gpus 1 \
    --finetune False \
    --num-workers 10 \
    --grad-accum 1 \
    --precision 32 \
    -o results/aligner/${method}_${lm}_finetune_mode_reg${reg_par}_500k

# note that unirep cannot be finetuned
