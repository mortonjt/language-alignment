import argparse
from language_alignment.benchmark.parse import msa2fasta
import glob
from functools import reduce
import os


def main(indir, out_sequences):
    folders = [x[0] for x in os.walk(indir)][1:]
    def getseqs(folder):
        alnf = glob.glob(f'{folder}/*.manual.ali')[0]
        s = tuple(msa2fasta(alnf))
        return s

    seqs = list(map(getseqs, folders))
    seqids = list(map(lambda x: os.path.basename(x), folders))
    with open(out_sequences, 'w') as out:
        for (id_, seq) in zip(seqids, seqs):
            seqidA = id_ + '.A'
            seqidB = id_ + '.B'
            seqA = seq[0]
            seqB = seq[1]
            out.write(f'>{seqidA}\n{seqA}\n')
            out.write(f'>{seqidB}\n{seqB}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract sequences')
    parser.add_argument('--indir', type=str, help='Input directory for mali benchmarks')
    parser.add_argument('--out-sequences', type=str, help='Output fasta file')
    args = parser.parse_args()
    main(args.indir, args.out_sequences)
