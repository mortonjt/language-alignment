#!/bin/bash
#

#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=16
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:v100-32gb:01

source ~/venvs/seqvec/bin/activate
module load slurm
module load cuda/10.0.130_410.48
module load cudnn/v7.6.2-cuda-10.0

DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
#in_file=$DATADIR/data/raw/combined.fasta
#in_file=$DATADIR/data/pfam/pfam_benchmark_seqs.fasta
#in_file=$DATADIR/data/raw/permuted.fasta
#results_dir=$DATADIR/results/embeddings/seqvec
#mkdir -p $results_dir

#python scripts/extract_seqvec.py $in_file $results_dir

# for f in $in_file
# do
#     seqvec -i $f -o $results_dir/embeddings.$f.npz --protein True --id -1
# done

in_file=/mnt/home/jmorton/ceph/seq-databases/pfam/families/PF.txt
results_dir=results/MSA/seqvec_embeds
mkdir -p $results_dir
seqvec -i $in_file -o $results_dir/embeddings.$f.npz --protein True --id -1
python scripts/extract_seqvec.py $in_file $results_dir
