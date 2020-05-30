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
#SBATCH -p genx

source ~/.bashrc
conda activate alignment

# DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
# inpath=$DATADIR/data/raw/swissprot-subset.fasta
# makeblastdb -in $inpath -dbtype 'prot' -out results/blast/swissprot
# blastp -db $DATADIR/results/blast/swissprot -query $inpath -out $DATADIR/results/blast/swissprot_benchmark.xml -outfmt 5
# pairs=$DATADIR/data/alignment/domain_pairs.txt
# blast_results=$DATADIR/results/blast/swissprot_benchmark.xml
# blast_alignments=$DATADIR/results/blast/swissprot_alignments.txt
# python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments
#
# pairs=$DATADIR/data/pfam/pairs.txt
# blast_results=$DATADIR/results/blast/swissprot_benchmark.xml
# blast_alignments=$DATADIR/results/blast/pfam_scores.txt
# python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments
#
# inpath=data/raw/scop_fa_represeq_lib20200117.fa
# makeblastdb -in $inpath -dbtype 'prot' -out results/blast/scop
# blastp -db results/blast/scop -query $inpath -out results/blast/scop_benchmark.xml -outfmt 5
# pairs=$DATADIR/data/scop/scop_pairs.txt
# blast_results=$DATADIR/results/blast/scop_benchmark.xml
# blast_alignments=$DATADIR/results/blast/scop_alignments.txt
# python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments
#
# inpath=$DATADIR/data/pfam/pfam_benchmark_seqs.fasta
# makeblastdb -in $inpath -dbtype 'prot' -out $DATADIR/results/blast/pfam
# blastp -db $DATADIR/results/blast/pfam -query $inpath -out $DATADIR/results/blast/pfam_benchmark.xml -outfmt 5
# pairs=$DATADIR/data/pfam/pairs.txt
# blast_results=$DATADIR/results/blast/pfam_benchmark.xml
# blast_alignments=$DATADIR/results/blast/pfam_alignments.txt
# python scripts/parse_blast_alignment.py $blast_results $pairs $blast_alignments


### PFAM / SCOP benchmarks
DATADIR=/mnt/home/jmorton/research/gert/icml2020/language-alignment
# inpath=/mnt/home/jmorton/research/gert/icml2020/language-alignment/data/scop
# inpath=$DATADIR/data/alignment-train/testing-set/test_pfam.fa
#
# mkdir -p $outdir
# makeblastdb -in $inpath -dbtype 'prot' -out $outdir
# blastp -db $outdir \
#     -query $inpath -out $outdir/pfam_benchmark.xml -outfmt 5
#
# inpath=$DATADIR/data/alignment-train/testing-set/test_scop.fa
# outdir=$DATADIR/results/distances/blast/scop
# mkdir -p $outdir
# makeblastdb -in $inpath -dbtype 'prot' -out $outdir
# blastp -db $outdir \
#     -query $inpath -out $outdir/scop_benchmark.xml -outfmt 5

# # parse results
echo 'pfam'
outdir=$DATADIR/results/distances/blast/pfam
blast_results=$DATADIR/results/distances/blast/pfam/pfam_benchmark.xml
blast_alignments=$DATADIR/results/distances/blast/pfam/pfam_alignments.txt
python scripts/parse_alignment.py \
    --path $blast_results \
    --aligner blast \
    --out-path $blast_alignments
python scripts/evaluate_blast.py \
    -p $DATADIR/data/alignment-train/testing-set/test_pfam.txt\
    -a $blast_alignments \
    -o $outdir/pfam_results.txt

echo 'scop'
outdir=$DATADIR/results/distances/blast/scop
blast_results=$DATADIR/results/distances/blast/scop/scop_benchmark.xml
blast_alignments=$DATADIR/results/distances/blast/scop/scop_alignments.txt
python scripts/parse_alignment.py \
    --path $blast_results \
    --aligner blast \
    --out-path $blast_alignments
python scripts/evaluate_blast.py \
    -p $DATADIR/data/alignment-train/testing-set/test_scop.txt\
    -a $blast_alignments \
    -o $outdir/scop_results.txt
