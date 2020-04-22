import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from scipy.optimize import linear_sum_assignment
from functools import reduce
import networkx as nx
import copy


def cca_solve(X, Y, n_components=10):
    """ Computes CCA """
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    r2 = cca.score(X, Y)
    U, V = cca.x_scores_, cca.y_scores_
    phi, psi = cca.x_loadings_, cca.y_loadings_
    return U, V, phi, psi, r2


def cca_align(phi, psi):
    """ Computes alignment based on canonical loadings

    This function computes a bipartite matching given the
    loadings.  Then Gaussian mixture models are constructed
    as our poor substitute for a null hypothesis.

    Parameters
    ----------
    phi : np.array
        Canonical loadings for X (columns are vectors)
    psi : np.array
        Canonical loadings for Y (columns are vectors)

    Returns
    -------
    cover_edges : np.array
        Full bipartite matching
    """
    cov_xy = psi @ np.linalg.pinv(phi)
    row_ind, col_ind = linear_sum_assignment(-cov_xy)
    cover_edges = pd.DataFrame({'source': row_ind, 'target': col_ind,
                                'weight': cov_xy[row_ind, col_ind]})
    return cover_edges

def filter_by_locality(cover, min_size=2):
    """ Filters residues based on locality

    If two adjacent residues in protein X map to adjacent residues in Y,
    then draw an edge between them. Only keep the connected components in this new graph
    that aren't doubletons.

    Parameters
    ----------
    cover_edges : pd.DataFrame
       Dataframe of edges with 'source', 'target' and 'weight' columns.
       Assumes that the edges are labelled according to row and column index.
    min_size :  int
       Minimum filtering criteria for a connected component.

    Returns
    -------
    filtered_edges : pd.DataFrame
       Filtered edges.
    """
    cover_edges = copy.copy(cover)
    cover_edges['source'] = list(map(lambda x: f'x{x}', cover_edges['source']))
    cover_edges['target'] = list(map(lambda x: f'y{x}', cover_edges['target']))

    tmpG = nx.from_pandas_edgelist(cover_edges)
    for i in range(1, cover_edges.shape[0]):
        prev_edge = cover_edges.iloc[i-1]
        edge = cover_edges.iloc[i]
        prev_xidx = int(prev_edge['source'][1:])
        cur_xidx = int(edge['source'][1:])
        prev_yidx = int(prev_edge['target'][1:])
        cur_yidx = int(edge['target'][1:])
        # if both are adjacent, draw edges
        if prev_xidx == (cur_xidx - 1) and prev_yidx == (cur_yidx - 1):
            tmpG.add_edge(prev_edge['source'], edge['source'])
            tmpG.add_edge(prev_edge['target'], edge['target'])

    components = list(nx.connected_components(tmpG))

    # filter out small components
    filtered_components = list(filter(lambda x: len(x) > min_size, components))
    nodes = reduce(lambda x, y: x | y, filtered_components)
    xnodes = list(filter(lambda x: x[0] == 'x', nodes))
    cover_edges = cover_edges.set_index('source')
    filtered_edges = cover_edges.loc[xnodes]
    idx = list(map(lambda x: int(x[1:]), filtered_edges.index))
    filtered_edges = filtered_edges.reset_index()

    filtered_edges['source'] = list(map(lambda x: int(x[1:]), filtered_edges.source))
    filtered_edges['target'] = list(map(lambda x: int(x[1:]), filtered_edges.target))
    return filtered_edges


def aln2edges(qseq: str, hseq: str):
    """
    Parameters
    ----------
    qseq : str
       Query sequence (with gaps)
    hseq : str
       Hit sequence (with gaps)

    Notes
    -----
    It is important to note that these sequences are
    assumed be the same length and aligned
    """
    assert len(qseq) == len(hseq)
    # convert to upper case
    qseq = qseq.upper()
    hseq = hseq.upper()
    q_coords = np.cumsum(np.array(list(qseq)) != '-')
    h_coords = np.cumsum(np.array(list(hseq)) != '-')
    edges = list(zip(list(q_coords), list(h_coords)))
    edges = list(map(lambda x: (x[3], x[4]), matches))
    return edges
