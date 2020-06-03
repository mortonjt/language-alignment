DATADIR=/mnt/home/jmorton/research/gert/icml2020/language-alignment

echo "PFam Distance"
ssa_pfam=`cat ${DATADIR}/results/aligner/ssa_model/pfam_results.txt | awk '$4 < $5 {print}' | wc -l`
cca_pfam=`cat ${DATADIR}/results/aligner/cca_model/pfam_results.txt | awk '$4 < $5 {print}' | wc -l`
total=`cat ${DATADIR}/results/aligner/ssa_model/pfam_results.txt | wc -l`
echo 'SSA' $ssa_pfam
echo 'CCA' $cca_pfam
echo 'Total' $total

echo "SCOP Distance"
ssa_scop=`cat ${DATADIR}/results/aligner/ssa_model/scop_results.txt | awk '$4 < $5 {print}' | wc -l`
cca_scop=`cat ${DATADIR}/results/aligner/cca_model/scop_results.txt | awk '$4 < $5 {print}' | wc -l`
total=`cat ${DATADIR}/results/aligner/ssa_model/scop_results.txt | wc -l`
echo 'SSA' $ssa_scop
echo 'CCA' $cca_scop
echo 'Total' $total
