from functools import reduce
import numpy as np
import pandas as pd
from language_alignment.alignment import cca_solve


def select_f(x):
    if x[2] != ' ':
        return x
    else:
        return None

def parse_hit(q_start: int, q_end:int, h_start:int, h_end:int,
              qseq: str, hseq: str, mseq: str):
    q_coords = np.cumsum(np.array(list(qseq)) != '-')
    h_coords = np.cumsum(np.array(list(hseq)) != '-')
    agg = list(zip(list(qseq), list(hseq), list(mseq),
                   list(q_coords), list(h_coords)))
    matches = list(map(select_f, agg))
    matches = list(filter(lambda x: x is not None, matches))
    edges = list(map(lambda x: (x[3], x[4]), matches))
    edges = pd.DataFrame(edges, columns=['source', 'target'])
    return edges

def interval_f(x, y):
    intv = np.arange(x, y)
    return set(intv)


def blast_hits(name, group):
    prot_x, prot_y, i = name
    #res = ground_truthing(prot_x, prot_y)
    #pfamx, pfamy, total_x, total_y = res

    xy = list(group.apply(
        lambda x: parse_hit(x['qs'], x['qe'], x['hs'], x['he'],
                            x['query_s'], x['hit_s'], x['aln_s']),
        axis=1
    ).values)

    xy = pd.concat(xy, axis=0)
    return prot_x, prot_y, xy


def domain_table(seq, dom):
    """
    Parameters
    ----------
    dom : pd.DataFrame
        Domain table

    Returns
    -------
    pd.DataFrame
        Per residue results, specifying if a residue
        belongs to a specific domain. Columns
        correspond to domains.
    """
    pos = np.arange(len(str(seq)))
    dpos = []
    for d in dom.domain.values:
        row = dom.loc[dom.domain == d]
        s, e = row['start'].values[0], row['end'].values[0]
        dpos.append(list(map(lambda x: is_interval(s, e, x), pos)))
    dpos = pd.DataFrame(dpos, index=dom.domain.values)

    # drop duplicates
    dpos = dpos.loc[~dpos.index.duplicated(keep='first')]
    return dpos.T


def domain_score(edges, seq1, dom1, seq2, dom2):
    """
    Parameters
    ----------
    seq : str
        Sequence 1
    dom : pd.DataFrame
        Domain table 1
    seq : str
        Sequence 2
    dom : pd.DataFrame
        Domain table 2

    Returns
    -------
    res : pd.DataFrame
        True positive and false positive results
    """
    df1 = domain_table(seq1, dom1)
    df2 = domain_table(seq2, dom2)
    res1 = pd.merge(edges, df1, left_on='source', right_index=True)
    res2 = pd.merge(edges, df2, left_on='target', right_index=True)
    resdf = pd.merge(res1, res2, left_on=['source', 'target'],
                     right_on=['source', 'target'], how='left')
    resdf = resdf.fillna(False)
    cols = list(set(dom1.domain.values) & set(dom2.domain.values))

    tps, fps, l = [], [], []
    for col in cols:
        colx = col + '_x'
        coly = col + '_y'
        tp = np.sum(np.logical_and(resdf[colx].values, resdf[coly].values))
        fp = np.sum(np.logical_and(resdf[colx].values, ~resdf[coly].values))
        tps.append(tp)
        fps.append(fp)
        l.append(df1[col].sum())

    res = pd.DataFrame({'tp': tps, 'fp': fps, 'len': l}, index=cols)
    return res


def score_alignment(pred_edges, truth_edges, total_length):
    """ Computes statistics for alignment.

    Parameters
    ----------
    pred_edges: list of tuples
       Predicted edges.
    truth_edges: list of tuples
       Ground truth edges.
    total_length : int
       Length of the alignment

    Returns
    -------
    tp, fp, tn, fn : ints
       True positive, false positive, true negatives and false negatives.
    """
    pred_edges = set(list(map(tuple, pred_edges)))
    truth_edges = set(list(map(tuple, truth_edges)))
    tp = len(set(pred_edges & truth_edges))
    fp = len(set(pred_edges - truth_edges))
    fn = len(set(truth_edges - pred_edges))
    # Compute the total number of possible
    total_edges = (total_length - 1) * total_length // 2
    tn = total_edges - tp - fp - fn
    return tp, fp, tn, fn


def score_group(group):
    prot_x, prot_y, edges = group
    dom_x = domdict[prot_x]
    dom_y = domdict[prot_y]
    sx = seqdict[prot_x]
    sy = seqdict[prot_y]
    res = domain_score(edges, sx, dom_x, sy, dom_y)
    tp = res.tp.sum()
    fp = res.fp.sum()
    l = res['len'].sum()
    return prot_x, prot_y, tp, fp, l


def is_interval(start, end, x):
    if x > start and x < end:
        return True
    return False


def distance(x, y, mode='euclidean', transpose=False):
    if transpose:
        X = x.T
        Y = y.T
    else:
        X = x.copy()
        Y = y.copy()

    if mode == 'euclidean':
        xc = X.mean(axis=0)
        yc = Y.mean(axis=0)
        return euclidean(xc, yc)

    elif mode == 'cca':
        r2 = cca_solve(X, Y, n_components=30)[-1]
        return 1 - r2
