import numpy as np
from Bio.Align import substitution_matrices
from Bio import Align
import pandas as pd


def init_aligner(dm=None):
    aligner = Align.PairwiseAligner()
    if dm is None:
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    else:
        aligner.substitution_matrix = Array(data=dm, dims=2, alphabet='ARNDCQEGHILKMFPSTWYVBZX*')

    return aligner


def pairwise_align(aligner, x, y):
    """
    Parameters
    ----------
    x : str
       Query sequence
    y : str
       Target sequence
    dm : np.array
       Embedding scores

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
