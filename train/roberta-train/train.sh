#!/bin/bash
#
#SBATCH --job-name=roberta
#SBATCH --output=stdout.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=40
#SBATCH --time=200:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# see https://www.glue.umd.edu/hpcc/help/software/pytorch.html#distrib
# may want to launch at pcn-7-12 

ip=`curl ifconfig.me`

module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
module load nccl/2.4.2-cuda-10.1
#export NCCL_DEBUG=WARN
#export NCCL_DEBUG_SUBSYS=ALL
#cd /simons/scratch/jmorton/mgt
cd /home/jmorton/research/gert/roberta-train
source ~/venvs/transformers-torch/bin/activate


# DIR=../data/pfam/train
# for SPLIT in train valid test; do \
#     python multiprocessing_bpe_encoder.py \
#         --encoder-json peptide_bpe/encoder.json \
#         --vocab-bpe peptide_bpe/vocab.bpe \
#         --inputs $DIR/Pfam-A.${SPLIT}.spaced.txt \
#         --outputs $DIR/Pfam-A.${SPLIT}.bpe \
#         --keep-empty \
#         --workers 40
# done
# 
# fairseq-preprocess \
#     --only-source \
#     --srcdict peptide_bpe/dict.txt \
#     --trainpref $DIR/Pfam-A.train.bpe \
#     --validpref $DIR/Pfam-A.valid.bpe \
#     --testpref $DIR/Pfam-A.test.bpe \
#     --destdir $DIR
#     --workers 40


DATA_DIR=../data/pfam/train/data-bin
SAVE_DIR=../data/pfam/checkpoints
TB_DIR=../data/pfam/tb

mkdir -p $SAVE_DIR
mkdir -p $TB_DIR

TOTAL_UPDATES=100000    # Total number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0009          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=1024  # Max sequence length
MAX_POSITIONS=1024      # Num. positional embeddings (usually same as above)
MAX_SENTENCES=4         # Number of sequences per batch (batch size)
UPDATE_FREQ=8           # Increase the batch size by fold
# DATA_DIR=/simons/scratch/jmorton/mgt/data/uniref100
# SAVE_DIR=/simons/scratch/jmorton/mgt/checkpoints/uniref100
# TB_DIR=/simons/scratch/jmorton/mgt/logdir/uniref100
PYTHON=/home/jmorton/venvs/roberta/bin/python
FAIRSEQ=/home/jmorton/venvs/roberta/bin/fairseq-train
# OMP_NUM_THREADS=10
echo `which python`

NPROC_PER_NODE=40
$(which fairseq-train) $DATA_DIR \
    --task masked_lm --criterion masked_lm \
    --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
    --ddp-backend=no_c10d \
    --arch roberta_base \
    --bpe gpt2 --memory-efficient-fp16 \
    --num-workers $NPROC_PER_NODE \
    --save-interval-updates 10000 \
    --save-dir $SAVE_DIR


