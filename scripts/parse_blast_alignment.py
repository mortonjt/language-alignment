from Bio import SearchIO
import numpy as np
import pandas as pd
import sys

# blast_path = '../results/blast_domain_benchmark.xml'
# dom_path = '../data/domains/common_pairs.txt'
# out_path = '../results/blast_domain_alignments.txt'

blast_path = sys.argv[1]
dom_path = sys.argv[2]
out_path = sys.argv[3]

domains = pd.read_table(dom_path, sep='\s+', header=None, dtype=str)
sets = domains.apply(lambda x: tuple(list(x)), axis=1)
domain_sets = set(list(sets.values))
records = SearchIO.parse(blast_path, 'blast-xml')
hit_list = []
with open(out_path, 'w') as out_handle:
    for idx, cur in enumerate(records):
        for hit in cur.hits:
            if (cur.id, hit.id) in domain_sets or (hit.id, cur.id) in domain_sets:
                for i, hsp in enumerate(hit.hsps):
                    print(cur.id, hit.id, i)
                    qs = hsp.fragment.query_start
                    qe = hsp.fragment.query_end
                    he = hsp.fragment.hit_end
                    hs = hsp.fragment.hit_start
                    query_s = str(hsp.fragment.query.seq)
                    hit_s = str(hsp.fragment.hit.seq)
                    aln_s = hsp.aln_annotation['similarity']
                    score = hsp.bitscore
                    expect = hsp.evalue
                    toks = map(str, [cur.id, hit.id, i, qs, qe, he, hs, query_s, hit_s, aln_s,
                                     score, expect])
                    line = '\t'.join(toks)
                    out_handle.write(f'{line}\n')
