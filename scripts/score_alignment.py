import os
import pandas as pd
import numpy as np
import glob
import site
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(os.path.join(base_path, 'src'))
from score import max_score, domain_score
import sys
from Bio import SeqIO

# print(sys.argv)
# in_directory = sys.argv[1]
# fasta_file = sys.argv[2]
# pair_file = sys.argv[3]
# dom_file = sys.argv[4]
# out_file = sys.argv[5]

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-i','--input-directory', help='Input directory', required=True)
parser.add_argument('-p','--pairs', help='PFam domain pairs', required=True)
parser.add_argument('-a','--alignments', help='Table of alignments', required=True)
parser.add_argument('-z','--zero-based', help='Zero based indexing or 1 based',
                    required=False, default=True, type=bool)
parser.add_argument('-o','--outdir', help='Output directory of edges', required=True)
args = parser.parse_args()

in_directory = args.input_directory
zero_based = args.zero_based
# in_directory = "../results/pw_elmo_domains/"
# fasta_file = "../../results/swissprot2.fasta"
# pair_file = "../data/domains/domain_pairs.txt"
# dom_file = "../data/domains/swissprot-pfam-domains.csv"
# out_file = "../results/pw_elmo_domain_scores.txt"

fnames = glob.glob(f'{in_directory}/*.csv')
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
#seqdict = {x.id : str(x.seq) for x in SeqIO.parse(fasta_file, "fasta")}
print(len(fnames), len(seqdict), len(domdict))
with open(out_file, 'w') as outhandle:
    for fname in fnames:
        print(fname, fname not in qsd, qsd[fname] not in set(pairs.index))
        if (fname not in qsd) and (qsd[fname] not in set(pairs.index)): continue
        if qsd[fname] not in pairs.index: continue

        prot_x, prot_y = pairs.loc[qsd[fname], 0], pairs.loc[qsd[fname], 1]
        dom_x, dom_y = domdict[prot_x], domdict[prot_y]
        edges_xy = pd.read_csv(fname, index_col=0)
        if zero_based:
            edges_xy = edges_xy + 1 # convert to 1 based indexing
        #seq_x, seq_y = seqdict[prot_x], seqdict[prot_y]
        res = domain_score(edges_xy, dom_x, dom_y)
        res = res.sum(axis=0)
        tp, fp, mx = res['tp'], res['fp'], res['len']
        line = f'{prot_x}\t{prot_y}\t{tp}\t{fp}\t{mx}\n'
        outhandle.write(line)
