import numpy as np
from Bio.Align import substitution_matrices
from Bio import Align
from scipy.spatial.distance import euclidean, cosine
import pandas as pd
import numba


def init_aligner(dm=None):
    aligner = Align.PairwiseAligner()
    if dm is None:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    else:
        aligner.substitution_matrix = Array(data=dm, dims=2, alphabet='ARNDCQEGHILKMFPSTWYVBZX*')

    return aligner



def smith_waterman_language_alignment(x, y, gap=0):
    """ Smith Waterman using language models.

    Parameters
    ----------
    x : np.array
       Embedding matrix for protein x
    y : np.array
       Embedding matrix for protein y
    gap : float
       Gap score

    Returns
    -------
    dm : np.array
       Alignment matrix
    score : float
       Alignment score
    """
    n = x.shape[0]
    m = y.shape[0]
    dm = np.zeros((n, m))
    tr = {}
    for i in range(1, n):
        for j in range(1, m):
            sij = 1 - cosine(x[i, :], y[j, :])
            entries = [dm[i-1, j-1] + sij,
                       dm[i-1, j] - gap,
                       dm[i, j-1] - gap]
            dm[i, j] = max(entries)
            k = np.argmax(entries)
            idx = [(i-1, j-1), (i-1, j), (i, j-1)]
            ti, tj = idx[k]
            tr[(i, j)] = (ti, tj)
    edges = traceback(tr, i, j, min(n, m))
    score = dm[-1, -1]
    return dm, score, edges

def traceback(tr, ti, tj, n):
    edges = [(ti, tj)]
    for _ in range(n):
        ti, tj = tr[(ti, tj)]
        edges.append((ti, tj))
    return edges

def pairwise_align(aligner, x, y):
    """
    Parameters
    ----------
    aligner : AlignmentModel

    x : str
       Query sequence
    y : str
       Target sequence

    Returns
    -------
    edges : list of tuples
       Edges in the alignment
    score : float
       Alignment score

    Notes
    -----
    If dm is not specified, then BLOSUM62 will be used for default.
    """
    try:
        alignments = aligner.align(x, y)
        optimal = next(alignments)
        edges = optimal.aligned
        score = optimal.score
    except:
        score = 10000
        edges = []
    return score, edges
