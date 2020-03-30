DATADIR=/mnt/ceph/users/jmorton/icml-final-submission
echo "PFam Distance"
epoch=5
attn_e=`cat ${DATADIR}/results/distances/attn/attn_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
attn_c=`cat ${DATADIR}/results/distances/attn/attn_epoch${epoch}_distances.txt | awk '$6 < $7 {print}' | wc -l`
attn90_e=`cat ${DATADIR}/results/distances/attn/attn_epochuniref90_distances.txt | awk '$4 < $5 {print}' | wc -l`
attn90_c=`cat ${DATADIR}/results/distances/attn/attn_epochuniref90_distances.txt | awk '$6 < $7 {print}' | wc -l`
elmo_e=`cat ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
elmo_c=`cat ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '$6 < $7 {print}' | wc -l`
total=`wc -l ${DATADIR}/results/distances/elmo/elmo_epoch${epoch}_distances.txt | awk '{print $1}'`
echo "${epoch} ${attn90_e} ${attn_e} ${elmo_e} ${total} Euclidean"
echo "${epoch} ${attn90_c} ${attn_c} ${elmo_c} ${total} Cosine"


echo "Domain Distance"
attn_e=`cat ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '$10 < $11 {print}' | wc -l`
elmo_e=`cat ${DATADIR}/results/domains/elmo/elmo_domain_epoch${epoch}.txt | awk '$10 < $11 {print}' | wc -l`
attn_c=`cat ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '$12 < $13 {print}' | wc -l`
elmo_c=`cat ${DATADIR}/results/domains/elmo/elmo_domain_epoch${epoch}.txt | awk '$12 < $13 {print}' | wc -l`
attn90_e=`cat ${DATADIR}/results/domains/attn/attn_domain_epochuniref90.txt | awk '$10 < $11 {print}' | wc -l`
attn90_c=`cat ${DATADIR}/results/domains/attn/attn_domain_epochuniref90.txt | awk '$12 < $13 {print}' | wc -l`
total=`wc -l ${DATADIR}/results/domains/attn/attn_domain_epoch${epoch}.txt | awk '{print $1}'`
echo "${epoch} ${attn90_e} ${attn_e} ${elmo_e} ${total} Euclidean"
echo "${epoch} ${attn90_c} ${attn_c} ${elmo_c} ${total} Cosine"

echo "Scop Distance"
attn_e=`cat ${DATADIR}/results/scop/attn/attn_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
elmo_e=`cat ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '$4 < $5 {print}' | wc -l`
attn_c=`cat ${DATADIR}/results/scop/attn/attn_epoch${epoch}_distances.txt | awk '$6 < $7 {print}' | wc -l`
elmo_c=`cat ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '$6 < $7 {print}' | wc -l`
attn90_e=`cat ${DATADIR}/results/scop/attn/attn_epochuniref90_distances.txt | awk '$4 < $5 {print}' | wc -l`
attn90_c=`cat ${DATADIR}/results/scop/attn/attn_epochuniref90_distances.txt | awk '$6 < $7 {print}' | wc -l`
total=`wc -l ${DATADIR}/results/scop/elmo/elmo_epoch${epoch}_distances.txt | awk '{print $1}'`
echo "${epoch} ${attn90_e} ${attn_e} ${elmo_e} ${total} Euclidean"
echo "${epoch} ${attn90_c} ${attn_c} ${elmo_c} ${total} Cosine"
