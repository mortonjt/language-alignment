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
#SBATCH -N 1

source ~/venvs/transformers-torch/bin/activate

conda activate alignment

DATADIR=/mnt/ceph/users/jmorton/icml-final-submission

inpath=$DATADIR/data/raw/swissprot-subset.fasta
makeblastdb -in $inpath -dbtype 'prot' -out results/blast/swissprot
blastp -db $DATADIR/results/blast/swissprot -query $inpath -out $DATADIR/results/blast/swissprot_benchmark.xml -outfmt 5
pairs=$DATADIR/data/alignment/domain_pairs.txt
blast_results=$DATADIR/results/blast/swissprot_benchmark.xml
blast_alignments=$DATADIR/results/blast/swissprot_alignments.txt
python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments

pairs=$DATADIR/data/pfam/pairs.txt
blast_results=$DATADIR/results/blast/swissprot_benchmark.xml
blast_alignments=$DATADIR/results/blast/pfam_scores.txt
python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments

inpath=data/raw/scop_fa_represeq_lib20200117.fa
makeblastdb -in $inpath -dbtype 'prot' -out results/blast/scop
blastp -db results/blast/scop -query $inpath -out results/blast/scop_benchmark.xml -outfmt 5
pairs=$DATADIR/data/scop/scop_pairs.txt
blast_results=$DATADIR/results/blast/scop_benchmark.xml
blast_alignments=$DATADIR/results/blast/scop_alignments.txt
python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments

inpath=$DATADIR/data/pfam/pfam_benchmark_seqs.fasta
makeblastdb -in $inpath -dbtype 'prot' -out $DATADIR/results/blast/pfam
blastp -db $DATADIR/results/blast/pfam -query $inpath -out $DATADIR/results/blast/pfam_benchmark.xml -outfmt 5
opairs=$DATADIR/data/pfam/pairs.txt
blast_results=$DATADIR/results/blast/pfam_benchmark.xml
blast_alignments=$DATADIR/results/blast/pfam_alignments.txt
python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments
