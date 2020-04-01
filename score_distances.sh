DATADIR=/mnt/ceph/users/jmorton/icml-final-submission

epoch=5


echo "Domain Distance"
bertBFD_e=`cat ${DATADIR}/results/domains/bert/BFD100_domain.txt | awk '$10 < $11 {print}' | wc -l`
bert100_e=`cat ${DATADIR}/results/domains/bert/Uniref100_domain.txt | awk '$10 < $11 {print}' | wc -l`
transformerXLBFD_e=`cat ${DATADIR}/results/domains/transformerXL/BFD100_domain.txt | awk '$10 < $11 {print}' | wc -l`
transformerXL100_e=`cat ${DATADIR}/results/domains/transformerXL/Uniref100_domain.txt | awk '$10 < $11 {print}' | wc -l`
attn_e=`cat ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '$10 < $11 {print}' | wc -l`
elmo_e=`cat ${DATADIR}/results/domains/elmo/elmo_domain_epoch${epoch}.txt | awk '$10 < $11 {print}' | wc -l`
attn90_e=`cat ${DATADIR}/results/domains/attn/attn_domain_epochuniref90.txt | awk '$10 < $11 {print}' | wc -l`
total=`wc -l ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '{print $1}'`
echo "${attn90_e} ${attn_e} ${elmo_e} ${bertBFD_e} ${bert100_e} ${transformerXLBFD_e} ${transformerXL100_e} ${total}"


echo "PFam Distance"
epoch=5
bertBFD_e=`cat ${DATADIR}/results/pfam/bert/BFD100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
bert100_e=`cat ${DATADIR}/results/pfam/bert/Uniref100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
transformerXLBFD_e=`cat ${DATADIR}/results/pfam/transformerXL/BFD100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
transformerXL100_e=`cat ${DATADIR}/results/pfam/transformerXL/Uniref100_pfam.txt | awk '$4 < $5 {print}' | wc -l`
attn_e=`cat ${DATADIR}/results/distances/attn/attn_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
attn90_e=`cat ${DATADIR}/results/distances/attn/attn_epochuniref90_distances.txt | awk '$4 < $5 {print}' | wc -l`
elmo_e=`cat ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
total=`wc -l ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '{print $1}'`
echo "${attn90_e} ${attn_e} ${elmo_e} ${bertBFD_e} ${bert100_e} ${transformerXLBFD_e} ${transformerXL100_e} ${total}"


echo "SCOP Distance"
epoch=5
bertBFD_e=`cat ${DATADIR}/results/scop/bert/BFD100_scop.txt | awk '$4 < $5 {print}' | wc -l`
bert100_e=`cat ${DATADIR}/results/scop/bert/Uniref100_scop.txt | awk '$4 < $5 {print}' | wc -l`
transformerXLBFD_e=`cat ${DATADIR}/results/scop/transformerXL/BFD100_scop.txt | awk '$4 < $5 {print}' | wc -l`
transformerXL100_e=`cat ${DATADIR}/results/scop/transformerXL/Uniref100_scop.txt | awk '$4 < $5 {print}' | wc -l`
attn_e=`cat ${DATADIR}/results/scop/attn/attn_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
elmo_e=`cat ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
attn90_e=`cat ${DATADIR}/results/scop/attn/attn_epochuniref90_distances.txt | awk '$4 < $5 {print}' | wc -l`
total=`wc -l ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '{print $1}'`
echo "${attn90_e} ${attn_e} ${elmo_e} ${bertBFD_e} ${bert100_e} ${transformerXLBFD_e} ${transformerXL100_e} ${total}"
