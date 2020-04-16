from Bio import SeqIO
import torch
import numpy as np
import sys
import pickle
import tensorflow as tf
import glob


# extract dimensions, average and save to disk
in_file = sys.argv[1]
model_path = sys.argv[2]
results_dir = sys.argv[3]
#in_file = '../results/subsetted_swissprot.fasta'
#out_file = '../results/subsetted_swissprot_encoded.pickle'
L = 768
N = 1022
#results_dir = '../results/swissprot-attn'
results = []


AMINOS = list("-DGULNTKHYWCPVSOIEFXQABZRM")
class ProteinSequence(object):
    """Sequence object for data manipulation"""
    SPECIAL_CHAR = '~'

    def __init__(self, seq: str):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __str__(self):
        return self.seq

    def __getitem(self, i):
        return self.seq[i]

    @property
    def onehot(self):
        """Map (fasta) sequence chars -> onehot vectors
        Special character "~" maps to the zero vector for padding
        """

        seq = list(self.seq)
        vocab_size = len(AMINOS)
        vocab_embed = dict(zip(AMINOS, range(vocab_size)))

        # Convert vocab to one-hot
        vocab_one_hot = np.zeros((vocab_size + 1, vocab_size), int)
        for _, val in vocab_embed.items():
            vocab_one_hot[val, val] = 1 # the ith amino becomes a onehot vector with position i filled
        vocab_embed[ProteinSequence.SPECIAL_CHAR] = vocab_size

        embed_x = [vocab_embed[v] for v in seq]
        seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])
        return seqs_x.reshape(1, *seqs_x.shape)

    def point_mutate(self, mut: str, position: int,
            zero_indexed:bool=False, ref:str=''):# -> ProteinSequence:
        """Performs a point mutation

        Supports point substitutions and deletions ('-')

        args:
            :mut      (str) - mutant amino
            :position (int) - mutation position
            :zero_indexed (bool) - flag `position` as zero or one-indexed
            :ref (str) - reference amino;
                         if specified, verify that `self.sequence[position]` == ref` prior to mutation
        returns:
            :(ProteinSequence) - mutated sequence
        raises:
            ValueError if `ref` has been specified and a mismatch was found
            ValueError if `ref not in AMINOS`
            IndexError if position is out of sequence bounds
        """
        seq      = list(self.seq)
        ref      = ref or None
        position = int(position - 1 if not zero_indexed else position)

        if ref not in [*AMINOS, None]:
            raise ValueError(f"{ref} is not a valid 'reference amino'")
        if ref is not None and self.seq[position] != ref:
            raise ValueError(f"Mismatched 'reference amino': expected {ref}, got {self.seq[position]}")
        try:
            seq[position] = mut
        except IndexError as e:
            raise IndexError(f"{position} is out of bounds for sequence of length {len(self)}") from e
        return ProteinSequence("".join(seq))

    def flank(self, center: int, window_size: int):#-> ProteinSequence:
        """
        Produces a 'flanking sequence' of length `window_size` centered on `center`.
        If the window goes outside of the sequence (i.e, window_size = 10 with center = 1),
        out of bounds positions are encoded with special character '~'

        The resulting sequence will be length (window_size - 1)/2 + 1 + (window_size - 1)/2

        args:
            :center (int) - center position of flanking sequence
            :window_size (int) - total window width, incremented if even
        returns:
            :(ProteinSequence) containing the flanking seq
        """
        if not window_size % 2: window_size += 1
        seq = self.seq
        flank_width = (window_size - 1) // 2
        start, stop = map(int, (center - flank_width, center + flank_width))
        flank = [seq[i] if 0 <= i <= len(self) - 1 else ProteinSequence.SPECIAL_CHAR for i in range(start, stop+1)]
        return ProteinSequence("".join(flank))

N = 1022


def create_feature_extractor(path, layer='LSTM2', trainable=False):
    m = tf.keras.models.load_model(path)
    return tf.keras.models.Model(inputs=[m.input],
                                 outputs=[m.get_layer(layer).output],
                                 trainable=trainable)

fxr = create_feature_extractor(model_path)
precomputed = set(glob.glob(f'{results_dir}/*.npz'))
with open(in_file, "rU") as input_handle:

    for record in SeqIO.parse(input_handle, "fasta"):
        s = str(record.seq)
        #if len(s) > N: continue
        if len(s) > N: # truncate
            s = s[:N]
        fname = f'{results_dir}/{record.id}.npz'
        if fname in precomputed:
            print(f'{fname} is already computed')
        else:
            print(record.id)
            prot = ProteinSequence(s)
            embed  = fxr.predict(prot.onehot).squeeze()
            np.savez_compressed(fname, embed=embed,
                                sequence=np.array(s))
