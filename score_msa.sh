#!/bin/bash
#
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p genx

source ~/venvs/transformers-torch/bin/activate
results_dir=/mnt/home/jmorton/research/gert/icml2020/language-alignment/results/MSA
embeds=$results_dir/elmo_embeds
metadata=$results_dir/msa_metadata.txt
out=$results_dir/elmo_counts_cca.txt
python scripts/score_msa.py $embeds $metadata $out False cca

# embeds=$results_dir/seqvec
# metadata=$results_dir/msa_metadata.txt
# out=$results_dir/seqvec_counts.txt
# python scripts/score_msa.py $embeds $metadata $out True

# results_dir=/mnt/home/jmorton/research/gert/icml2020/language-alignment/results/MSA
# embeds=~/ceph/embeddings/pfam-families/roberta-xs
# metadata=$results_dir/msa_metadata.txt
# out=$results_dir/roberta_xs_counts_cca.txt
# python scripts/score_msa.py $embeds $metadata $out False cca
#
# results_dir=/mnt/home/jmorton/research/gert/icml2020/language-alignment/results/MSA
# embeds=~/ceph/embeddings/pfam-families/roberta-gert
# metadata=$results_dir/msa_metadata.txt
# out=$results_dir/roberta_gert_counts_cca.txt
# python scripts/score_msa.py $embeds $metadata $out False cca
