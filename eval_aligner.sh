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
lm=roberta
for method in ssa cca
do
    for dataset in pfam scop
    do
        test_file=$datadir/testing-set/test_${dataset}.txt
        fasta_file=$datadir/seqs.fasta
        # results_dir=results/aligner
        model=/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert
        # model=/mnt/home/jmorton/research/gert/icml2020/language-alignment
        echo $method
        python scripts/evaluate_aligner.py \
            --test-pairs $test_file \
            --fasta $fasta_file \
            -m $model \
            --arch 'roberta' \
            --aligner $method \
            --model-path $model/results/aligner/${method}_${lm}_finetune_model \
            --gpu True \
            --lm-embed-dim 1024 \
            --aligner-embed-dim 1024 \
            -o results/aligner/${method}_model/${dataset}_results.txt
    done
done
