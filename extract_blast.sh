#!/bin/bash
#
#SBATCH --job-name=blast
#SBATCH --output=stdout_blast.txt
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#
#SBATCH --ntasks=4
#SBATCH --time=30:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH -p genx
#SBATCH -N 1
source ~/.bashrc
conda activate alignment

DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
alignments=$DATADIR/results/blast/swissprot_alignments.txt
fasta=~/ceph/icml-final-submission/data/raw/combined.fasta
pairs=~/ceph/icml-final-submission/data/alignment/domain_pairs.txt
domains=~/ceph/icml-final-submission/data/alignment/swissprot-pfam-domains.csv
results=$DATADIR/results/blast/edges
mkdir -p $results
python scripts/format_blast.py -a $alignments -p $pairs -d $domains -o $results
