DATADIR=/mnt/home/jmorton/research/gert/icml2020/language-alignment

echo "PFam distance"
roberta_ssa=$DATADIR/results/aligner/ssa_roberta_finetune_mode_reg1e-3_500k_round3
roberta_ssa=`cat $DATADIR/results/aligner/ssa_roberta_finetune_mode_reg1e-3_500k_round3/pfam_results.txt | awk '$4 > $5 {print}' | wc -l`
roberta_cca=`cat $DATADIR/results/aligner/cca_roberta_finetune_mode_reg1e-3_500k_round3/pfam_results.txt | awk '$4 < $5 {print}' | wc -l`
hmmer=`cat $DATADIR/results/distances/hmmer/pfam/pfam_results.txt | awk '$4 < $5 {print}' | wc -l`
blast=`cat $DATADIR/results/distances/blast/pfam/pfam_results.txt | awk '$4 < $5 {print}' | wc -l`
sw=`cat $DATADIR/results/distances/sw/pfam/pfam_results.txt | awk '$4 > $5 {print}' | wc -l`
echo 'Roberta SSA:' $roberta_ssa
echo 'Roberta CCA:' $roberta_cca
echo 'HMMER:' $hmmer
echo 'Blast:' $blast
echo 'Smith-Waterman:' $sw

echo "SCop distance"
roberta_ssa=$DATADIR/results/aligner/ssa_roberta_finetune_mode_reg1e-3_500k_round3
roberta_ssa=`cat $DATADIR/results/aligner/ssa_roberta_finetune_mode_reg1e-3_500k_round3/scop_results.txt | awk '$4 > $5 {print}' | wc -l`
roberta_cca=`cat $DATADIR/results/aligner/cca_roberta_finetune_mode_reg1e-3_500k_round3/scop_results.txt | awk '$4 < $5 {print}' | wc -l`
hmmer=`cat $DATADIR/results/distances/hmmer/scop/scop_results.txt | awk '$4 < $5 {print}' | wc -l`
blast=`cat $DATADIR/results/distances/blast/scop/scop_results.txt | awk '$4 > $5 {print}' | wc -l`
sw=`cat $DATADIR/results/distances/sw/scop/scop_results.txt | awk '$4 > $5 {print}' | wc -l`
echo 'Roberta SSA:' $roberta_ssa
echo 'Roberta CCA:' $roberta_cca
echo 'HMMER:' $hmmer
echo 'Blast:' $blast
echo 'Smith-Waterman:' $sw

# epoch=5
#
# echo "Domain Distance"
# bertBFD_e=`cat ${DATADIR}/results/domains/bert/BFD100_domain.txt | awk '$10 < $11 {print}' | wc -l`
# bert100_e=`cat ${DATADIR}/results/domains/bert/Uniref100_domain.txt | awk '$10 < $11 {print}' | wc -l`
# transformerXLBFD_e=`cat ${DATADIR}/results/domains/transformerXL/BFD100_domain.txt | awk '$10 < $11 {print}' | wc -l`
# transformerXL100_e=`cat ${DATADIR}/results/domains/transformerXL/Uniref100_domain.txt | awk '$10 < $11 {print}' | wc -l`
# seqvec_e=`cat ${DATADIR}/results/domains/seqvec/domain.txt | awk '$10 < $11 {print}' | wc -l`
# attn_e=`cat ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '$10 < $11 {print}' | wc -l`
# elmo_e=`cat ${DATADIR}/results/domains/elmo/elmo_domain_epoch${epoch}.txt | awk '$10 < $11 {print}' | wc -l`
# attn90_e=`cat ${DATADIR}/results/domains/attn/attn_domain_epochuniref90.txt | awk '$10 < $11 {print}' | wc -l`
# total=`wc -l ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '{print $1}'`
# echo "${attn90_e} ${attn_e} ${elmo_e} ${bertBFD_e} ${bert100_e} ${transformerXLBFD_e} ${transformerXL100_e} ${seqvec_e} ${total}"
#
#
# echo "PFam Distance"
# epoch=5
# bertBFD_e=`cat ${DATADIR}/results/pfam/bert/BFD100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
# bert100_e=`cat ${DATADIR}/results/pfam/bert/Uniref100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
# transformerXLBFD_e=`cat ${DATADIR}/results/pfam/transformerXL/BFD100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
# transformerXL100_e=`cat ${DATADIR}/results/pfam/transformerXL/Uniref100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
# seqvec_e=`cat ${DATADIR}/results/pfam/seqvec/pfam.txt | awk '$4 < $5 {print}' | wc -l`
# attn_e=`cat ${DATADIR}/results/distances/attn/attn_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
# attn90_e=`cat ${DATADIR}/results/distances/attn/attn_epochuniref90_distances.txt | awk '$4 < $5 {print}' | wc -l`
# elmo_e=`cat ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
# total=`wc -l ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '{print $1}'`
# echo "${attn90_e} ${attn_e} ${elmo_e} ${bertBFD_e} ${bert100_e} ${transformerXLBFD_e} ${transformerXL100_e} ${seqvec_e} ${total}"
#
#
#
# echo "SCOP Distance"
# epoch=5
# bertBFD_e=`cat ${DATADIR}/results/scop/bert/BFD100_scop.txt | awk '$4 < $5 {print}' | wc -l`
# bert100_e=`cat ${DATADIR}/results/scop/bert/Uniref100_scop.txt | awk '$4 < $5 {print}' | wc -l`
# transformerXLBFD_e=`cat ${DATADIR}/results/scop/transformerXL/BFD100_scop.txt | awk '$4 < $5 {print}' | wc -l`
# transformerXL100_e=`cat ${DATADIR}/results/scop/transformerXL/Uniref100_scop.txt | awk '$4 < $5 {print}' | wc -l`
# seqvec_e=`cat ${DATADIR}/results/scop/seqvec/scop.txt | awk '$4 < $5 {print}' | wc -l`
# attn_e=`cat ${DATADIR}/results/scop/attn/attn_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
# elmo_e=`cat ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
# attn90_e=`cat ${DATADIR}/results/scop/attn/attn_epochuniref90_distances.txt | awk '$4 < $5 {print}' | wc -l`
# total=`wc -l ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '{print $1}'`
# echo "${attn90_e} ${attn_e} ${elmo_e} ${bertBFD_e} ${bert100_e} ${transformerXLBFD_e} ${transformerXL100_e} ${seqvec_e} ${total}"
