# see https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.pretraining.md
#DIR=/simons/scratch/jmorton/uniprot/uniref100

# for SPLIT in train valid test; do \
#     python multiprocessing_bpe_encoder.py \
#         --encoder-json peptide_bpe/encoder.json \
#         --vocab-bpe peptide_bpe/vocab.bpe \
#         --inputs $DIR/${SPLIT}_shuffle.txt \
#         --outputs data/${SPLIT}.uniref100_trimmed.bpe \
#         --keep-empty \
#         --workers 10
# done
# 
# fairseq-preprocess \
#     --only-source \
#     --srcdict peptide_bpe/dict.txt \
#     --trainpref data/train.uniref100_trimmed.bpe \
#     --validpref data/valid.uniref100_trimmed.bpe \
#     --testpref data/test.uniref100_trimmed.bpe \
#     --destdir data/uniref100 \
#     --workers 10



DIR=../data/pfam/train
for SPLIT in train valid test; do \
    python multiprocessing_bpe_encoder.py \
        --encoder-json peptide_bpe/encoder.json \
        --vocab-bpe peptide_bpe/vocab.bpe \
        --inputs $DIR/Pfam-A.${SPLIT}.spaced.txt \
        --outputs $DIR/Pfam-A.${SPLIT}.bpe \
        --keep-empty \
        --workers 1
done

# fairseq-preprocess \
#     --only-source \
#     --srcdict peptide_bpe/dict.txt \
#     --trainpref $DIR/Pfam-A.train.bpe \
#     --validpref $DIR/Pfam-A.valid.bpe \
#     --testpref $DIR/Pfam-A.test.bpe \
#     --workers 1
