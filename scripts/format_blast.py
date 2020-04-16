import argparse
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(os.path.join(base_path, 'src'))
from score import blast_hits


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-d','--domains', help='PFam domains', required=True)
parser.add_argument('-p','--pairs', help='PFam domain pairs', required=True)
parser.add_argument('-a','--alignments', help='Table of alignments', required=True)
parser.add_argument('-o','--outdir', help='Output directory of edges', required=True)
args = parser.parse_args()

# Read in files
dom_file = args.domains
dom_pairs = args.pairs
pairs = pd.read_table(dom_pairs, header=None, sep='\s+')
df = pd.read_csv(dom_file, header=None, skiprows=1)
df.columns = ['protein', 'domain', 'source',
              'domain_id', 'start', 'end']
df['length'] = df.apply(lambda x: x['end'] - x['start'], axis=1)
domdict = dict(list(df.groupby('protein')))
blast_df = pd.read_table(args.alignments, header=None)
blast_df.columns = [
    'cur.id', 'hit.id', 'i',
    'qs', 'qe', 'he', 'hs',
    'query_s', 'hit_s', 'aln_s',
    'bitscore', 'evalue'
]
blast_df = blast_df.dropna()
blast_groups = blast_df.groupby(['cur.id','hit.id','i'])
res = list(map(lambda x: blast_hits(*x), blast_groups))
for query, target, df in res:
    fname = f'attn_edges_{query}{target}.csv'
    df.to_csv(f'{args.outdir}/{fname}')
