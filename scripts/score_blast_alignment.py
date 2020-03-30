import os
import pandas as pd
import numpy as np
import glob
import site
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(os.path.join(base_path, 'src'))
from score import max_score, domain_score, blast_hits
import sys
from Bio import SeqIO

print(sys.argv)
blast_file = sys.argv[1]
fasta_file = sys.argv[2]
pair_file = sys.argv[3]
dom_file = sys.argv[4]
out_file = sys.argv[5]

# in_directory = "../results/pw_elmo_domains/"
# fasta_file = "../../results/swissprot2.fasta"
# pair_file = "../data/domains/domain_pairs.txt"
# dom_file = "../data/domains/swissprot-pfam-domains.csv"
# out_file = "../results/pw_elmo_domain_scores.txt"

def score_group(group):
    prot_x, prot_y, edges = group
    dom_x = domdict[prot_x]
    dom_y = domdict[prot_y]
    sx = seqdict[prot_x]
    sy = seqdict[prot_y]
    res = domain_score(edges, sx, dom_x, sy, dom_y)
    tp = res.tp.sum()
    fp = res.fp.sum()
    return prot_x, prot_y, tp, fp

blast_df = pd.read_table(blast, header=None)
pairs = pd.read_csv(pair_file, header=None, sep='\s+')
pairs['name'] = pairs.apply(lambda x: f'attn_edges_{x[0]}{x[1]}.csv', axis=1)
pairs = pairs.set_index('name')
pairs = pairs.loc[~pairs.index.duplicated(keep='first')]
pairs_set = set(pairs.index)

qsd = list(map(os.path.basename, fnames))
qsd = dict(zip(fnames, qsd))
pairs_set = set(pairs.index)
print(len(pairs_set & set(qsd.keys())))

df = pd.read_csv(dom_file, header=None, skiprows=1)
df.columns = ['protein', 'domain', 'source',
              'domain_id', 'start', 'end']
df['length'] = df.apply(lambda x: x['end'] - x['start'], axis=1)
domdict = dict(list(df.groupby('protein')))

# load fasta seqs
seqdict = {x.id : str(x.seq) for x in SeqIO.parse(fasta_file, "fasta")}
print(len(fnames), len(seqdict), len(domdict))


blast_df.columns = [
    'cur.id', 'hit.id', 'i',
    'qs', 'qe', 'he', 'hs',
    'query_s', 'hit_s', 'aln_s',
    'bitscore', 'evalue'
]

blast_df = blast_df.dropna()

blast_groups = blast_df.groupby(['cur.id','hit.id','i'])
res = list(map(lambda x: blast_hits(*x), blast_groups))
stats = list(map(score_group, res))
blast_stats = pd.DataFrame(stats)
blast_stats.to_csv(out_file, sep='\t', header=None, index=None)
