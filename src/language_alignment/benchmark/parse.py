import os
from Bio.PDB import *


def pdb2seq(protein_file):
    """ Parses PDB file and returns sequence.

    Parameters
    ----------
    protein_file : filepath
        Path to pdb file

    Returns
    -------
    seq : str
       Protein sequence.
    seqid : str
       Sequence identifier.
    """
    parser = PDBParser()
    ppb = PPBuilder()
    seqs = []
    structure = parser.get_structure('', protein_file)
    for pp in ppb.build_peptides(structure):
        seqs.append(str(pp.get_sequence()))
    seq = ''.join(seqs)
    fname = os.path.basename(protein_file).split('.')[0]
    return seq, seqid


def msa2fasta(fname, format=None):
    """ Parser for multiple sequence alignments

    Parameters
    ----------
    fname : filepath
        Filepath to Multiple sequence alignment.
    format: str
        Specifies what file format the MSA is in.
        If this isn't specified, it will assume that
        each line is a unique sequence.

    Returns
    -------
    list of str
        List of sequences.
    """
    if format is None:
        seqs = open(fname).readlines()
        seqs = list(map(lambda x: x.rstrip(), seqs))
        seqs = list(map(lambda x: x.replace('-', ''), seqs))
        seqs = list(map(lambda x: x.upper(), seqs))
    else:
        raise ValueError(f'Format {format} is not supported')
    return seqs

