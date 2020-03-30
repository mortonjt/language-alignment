#!/bin/bash
#
#SBATCH --job-name=elmo_scop
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jmorton@flatironinstitute.org
#SBATCH --time=60:00:00
#SBATCH -N 1
#SBATCH -p ccb


module load cuda/10.1.105_418.39
module load cudnn/v7.6.2-cuda-10.1
source ~/venvs/transformers-torch/bin/activate
echo `which python`

mkdir -p results/scop/elmo
for epoch in 1 2 3 4 5
do
    embeds=results/embeddings/elmo/epoch${epoch}
    triples=data/scop/scop_triples.txt
    out=results/scop/elmo/elmo_epoch${epoch}_distances.txt
    python scripts/get_distances.py $embeds $triples $out
    echo "Epoch $epoch is done"
done
