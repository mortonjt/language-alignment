from Bio import SearchIO
import pandas as pd
import argparse
import sys

# blast_path = '../results/blast_domain_benchmark.xml'
# dom_path = '../data/domains/common_pairs.txt'
# out_path = '../results/blast_domain_alignments.txt'
parser = argparse.ArgumentParser(description='Parse alignments')
parser.add_argument('--path', help='Path to alignment output', required=True)
parser.add_argument('--out-path', help='Output path', required=True)
parser.add_argument('--aligner', help='Type of alignment (blast or hmmer)', required=True)
args = parser.parse_args()

if args.aligner == 'blast':
    fmt = 'blast-xml'
elif args.aligner == 'hmmer':
    fmt = 'hmmer3-text'
else:
    raise ValueError(f'Unsupported aligner {aligner}')

records = SearchIO.parse(args.path, fmt)
hit_list = []
with open(args.out_path, 'w') as out_handle:
    for idx, cur in enumerate(records):
        for hit in cur.hits:
            for i, hsp in enumerate(hit.hsps):
                qs = hsp.fragment.query_start
                qe = hsp.fragment.query_end
                he = hsp.fragment.hit_end
                hs = hsp.fragment.hit_start
                query_s = str(hsp.fragment.query.seq)
                hit_s = str(hsp.fragment.hit.seq)
                aln_s = None
                if 'similarity' in hsp.aln_annotation:
                    aln_s = hsp.aln_annotation['similarity']
                score = hsp.bitscore
                expect = hsp.evalue
                toks = map(str, [cur.id, hit.id, i, qs, qe, he, hs, query_s, hit_s, aln_s,
                                 score, expect])
                line = '\t'.join(toks)
                out_handle.write(f'{line}\n')
