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
homedir=/mnt/home/jmorton/research/gert/icml2020/language-alignment
datadir=$homedir/data/alignment-train
lm=roberta
method=ssa
dataset=pfam
# for dataset in pfam scop
# do

model=/mnt/home/jmorton/ceph/checkpoints/pfam/checkpoint_gert
# model=/mnt/home/jmorton/research/gert/icml2020/language-alignment
dataset=malisam
aliname=d1adja2d1drw_2
$homedir/results/aligner/${method}_${lm}_finetune_mode_reg1e-3_500k_round3
ali=~/ceph/seq-databases/structures/$dataset/$aliname/$aliname.manual.ali
echo $ali
results_dir=$homedir/results/struct_alignments/$dataset/$aliname/${method}_${lm}_finetune_mode_reg1e-3_500k_round3
mkdir -p $results_dir
echo $method
python scripts/evaluate_structural_alignment.py \
    --manual-alignment $ali \
    -m $model \
    --arch 'roberta' \
    --aligner $method \
    --model-path $results_dir \
    --gpu True \
    --lm-embed-dim 1024 \
    --aligner-embed-dim 1024 \
    --output-dir $results_dir
# done
