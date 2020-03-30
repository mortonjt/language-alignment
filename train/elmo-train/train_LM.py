import json
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
from keras.layers import Dense, GRU, CuDNNLSTM
# from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Sequential
# from keras.layers import Concatenate
from keras.constraints import MinMaxNorm
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from progressbar import ProgressBar
from sklearn.model_selection import KFold

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_protein
from Bio.SeqRecord import SeqRecord


plt.switch_backend('agg')
sess = tf.Session()
K.set_session(sess)

flags = tf.app.flags

flags.DEFINE_string("dataset", "training_sequences_noC.csv", "dataset file (expecting csv)")
flags.DEFINE_string("name", "test", "run name for log and checkpoint files")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("epochs", 50, "epochs to train")
flags.DEFINE_integer("layers", 2, "number of layers in the network")
flags.DEFINE_integer("neurons", 512, "number of units per layer")
flags.DEFINE_string("cell", "LSTM", "type of neuron to use, available: LSTM, GRU")
flags.DEFINE_float("dropout", 0.2, "dropout to use in every layer; layer 1 gets 1*dropout, layer 2 2*dropout etc.")
flags.DEFINE_boolean("train", True, "whether the network should be trained or just sampled from")
flags.DEFINE_float("valsplit", 0.2, "fraction of the data to use for validation")
flags.DEFINE_integer("sample", 2000, "number of sequences to sample training")
flags.DEFINE_float("temp", 2.5, "temperature used for sampling")
flags.DEFINE_integer("maxlen", 1000, "maximum sequence length allowed when sampling new sequences")
flags.DEFINE_string("startchar", "B", "starting character to begin sampling. Default='B' for 'begin'")
flags.DEFINE_float("lr", 0.005, "learning rate to be used with the Adam optimizer")
flags.DEFINE_float("l2", None, "l2 regularization rate. If None, no l2 regularization is used")
flags.DEFINE_string("modfile", None, "filename of the pretrained model to used for sampling if train=False")
flags.DEFINE_boolean("finetune", False, "if True, a pretrained model provided in modfile is finetuned on the dataset")
flags.DEFINE_integer("cv", None, "number of folds to use for cross-validation; if None, no CV is performed")
flags.DEFINE_integer("window", 0, "window size used to process sequences. If 0, all sequences are padded to the "
                                  "longest sequence length in the dataset")
flags.DEFINE_integer("step", 1, "step size to move window or prediction target")
flags.DEFINE_string("target", "all", "whether to learn all proceeding characters or just the last `one` in sequence")
flags.DEFINE_integer("padlen", 0, "number of spaces to use for padding sequences (if window not 0); if 0, sequences are"
                                  " padded to the length of the longest sequence in the dataset")
flags.DEFINE_boolean("refs", True, "whether reference sequence sets should be generated for the analysis")

FLAGS = flags.FLAGS


def _save_flags(filename):
    """ Function to save used tf.FLAGS to log-file

    :return: saved file
    """
    with open(filename, 'w') as f:
        f.write("Used flags:\n-----------\n")
        for k, v in tf.flags.FLAGS.__flags.items():
            f.write(k + ": " + str(v.value) + "\n")


def _onehotencode(s, vocab=None):
    """ Function to one-hot encode a sring.

    :param s: {str} String to encode in one-hot fashion
    :param vocab: vocabulary to use fore encoding, if None, default AAs are used
    :return: one-hot encoded string as a np.array
    """
    if not vocab:
        vocab = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P', 'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']

    # generate translation dictionary for one-hot encoding
    to_one_hot = dict()
    for i, a in enumerate(vocab):
        v = np.zeros(len(vocab))
        v[i] = 1
        to_one_hot[a] = v

    result = []
    for l in s:
        result.append(to_one_hot[l])
    result = np.array(result)
    return result, to_one_hot, vocab
    # return np.reshape(result, (1, result.shape[0], result.shape[1])), to_one_hot, vocab


def _onehotdecode(matrix, vocab=None, filename=None):
    """ Decode a given one-hot represented matrix back into sequences

    :param matrix: matrix containing sequence patterns that are one-hot encoded
    :param vocab: vocabulary, if None, standard AAs are used
    :param filename: filename for saving sequences, if ``None``, sequences are returned in a list
    :return: list of decoded sequences in the range lenmin-lenmax, if ``filename``, they are saved to a file
    """
    if not vocab:
        _, _, vocab = _onehotencode('A')
    if len(matrix.shape) == 2:  # if a matrix containing only one string is supplied
        result = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                aa = np.where(matrix[i, j] == 1.)[0][0]
                result.append(vocab[aa])
        seq = ''.join(result)
        if filename:
            with open(filename, 'wb') as f:
                f.write(seq)
        else:
            return seq

    elif len(matrix.shape) == 3:  # if a matrix containing several strings is supplied
        result = []
        for n in range(matrix.shape[0]):
            oneresult = []
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[2]):
                    aa = np.where(matrix[n, i, j] == 1.)[0][0]
                    oneresult.append(vocab[aa])
            seq = ''.join(oneresult)
            result.append(seq)
        if filename:
            with open(filename, 'wb') as f:
                for s in result:
                    f.write(s + '\n')
        else:
            return result


def _sample_with_temp(preds, temp=1.0):
    """ Helper function to sample one letter from a probability array given a temperature.

    :param preds: {np.array} predictions returned by the network
    :param temp: {float} temperature value to sample at.
    """
    streched = np.log(preds) / temp
    stretched_probs = np.exp(streched) / np.sum(np.exp(streched))
    return np.random.choice(len(streched), p=stretched_probs)


def load_model_instance(filename):
    """ Load a whole Model class instance from a given epoch file

    :param filename: epoch file, e.g. model_epoch_5.hdf5
    :return: model instance with trained weights
    """
    modfile = os.path.dirname(filename) + '/model.p'
    mod = pickle.load(open(modfile, 'rb'))
    hdf5_file = ''.join(modfile.split('.')[:-1]) + '.hdf5'
    mod.model = load_model(hdf5_file)
    return mod


def save_model_instance(mod):
    """ Save a whole Model instance and the corresponding model with weights to two files (model.p and model.hdf5)

    :param mod: model instance
    :return: saved model files in the checkpoint dir
    """
    tmp = mod.model
    tmp.save(mod.checkpointdir + 'model.hdf5')
    mod.model = None
    pickle.dump(mod, open(mod.checkpointdir + 'model.p', 'wb'))
    mod.model = tmp


class DataGenerator(keras.utils.Sequence):
    "### Generates data for Keras"
    def __init__(self, sequences, n_channels=26, batch_size=64, shuffle=True):
        "### Initialization"
        self.sequences = sequences
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "### Denotes the number of batches per epoch"
        return int(np.floor(len(self.sequences) / self.batch_size))

    def __getitem__(self, index):
        "### Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        sequences_temp = [self.sequences[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(sequences_temp)

        return X, y

    def on_epoch_end(self):
        "### Updates indexes after each epoch"
        self.indexes = np.arange(len(self.sequences))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, sequences_temp):
        "### Generates data containing batch_size samples"
        max_length = len(sequences_temp[0])

        X = np.zeros((self.batch_size, max_length-1, self.n_channels))
        y = np.zeros((self.batch_size, max_length-1, self.n_channels))
        for i, sequence in enumerate(sequences_temp):
            X[i], _, _ = _onehotencode(sequence[:-1])
            y[i], _, _ = _onehotencode(sequence[1:])

        return X, y


class SequenceHandler(object):
    """ Class for handling peptide sequences, e.g. loading, one-hot encoding or decoding and saving """

    def __init__(self, window=0, step=2, refs=True):
        """
        :param window: {str} window used for chopping up sequences. If 0: False
        :param step: {int} size of the steps to move the window forward
        :param refs {bool} whether to generate reference sequence sets for analysis
        """
        self.sequences = None
        self.generated = None
        # self.ran = None
        # self.hel = None
        self.X = list()
        self.y = list()
        self.window = window
        self.step = step
        self.refs = refs
        # generate translation dictionary for one-hot encoding
        _, self.to_one_hot, self.vocab = _onehotencode('A')

    def load_sequences(self, filename):
        """ Method to load peptide sequences from a csv file

        :param filename: {str} filename of the sequence file to be read (``csv``, one sequence per line)
        :return: sequences in self.sequences
        """
        with open(filename) as f:
            self.sequences = [s.strip() for s in f]
        self.sequences = random.sample(self.sequences, len(self.sequences))  # shuffle sequences randomly

    def load_fasta(self, filename):
        """ Method to load sequences from fasta file

        :param fasta filename
        :return: sequences in self.sequences
        """
        self.sequences = [str(entry.seq) for entry in SeqIO.parse(filename, 'fasta')]
        self.sequences = random.sample(self.sequences, len(self.sequences))  # shuffle sequences randomly

    def pad_sequences(self, pad_char='-', padlen=0):
        """ Pad all sequences to the longest length (default, padlen=0) or a given length

        :param pad_char: {str} Character to pad sequences with
        :param padlen: {int} Custom length of padding to add to all sequences to (optional), default: 0. If
        0, sequences are padded to the length of the longest sequence in the training set. If a window is used and the
        padded sequence is shorter than the window size, it is padded to fit the window.
        """
        if padlen:
            padded_seqs = []
            for seq in self.sequences:
                if len(seq) < self.window:
                    padded_seq = seq + pad_char * (self.step + self.window - len(seq))
                else:
                    padded_seq = seq + pad_char * padlen
                padded_seqs.append(padded_seq)
        else:
            length = max([len(seq) for seq in self.sequences])
            padded_seqs = []
            for seq in self.sequences:
                padded_seq = 'B' + seq + pad_char * (length - len(seq))
                padded_seqs.append(padded_seq)

        if pad_char not in self.vocab:
            self.vocab += [pad_char]

        self.sequences = padded_seqs  # overwrite sequences with padded sequences

    def one_hot_encode(self, target='all'):
        """ Chop up loaded sequences into patterns of length ``window`` by moving by stepsize ``step`` and translate
        them with a one-hot vector encoding

        :param target: {str} whether all proceeding AA should be learned or just the last one in sequence (`all`, `one`)
        :return: one-hot encoded sequence patterns in self.X and corresponding target amino acids in self.y
        """
        if self.window == 0:
            split = int(0.8*len(self.sequences))
            # train/validation
            self.X = DataGenerator(self.sequences[:split])
            self.y = DataGenerator(self.sequences[split:])
            """
            for s in self.sequences:
                self.X.append([self.to_one_hot[char] for char in s[:-self.step]])
                if target == 'all':
                    self.y.append([self.to_one_hot[char] for char in s[self.step:]])
                elif target == 'one':
                    self.y.append(s[-self.step:])

            self.X = np.reshape(self.X, (len(self.X), len(self.sequences[0]) - self.step, len(self.vocab)))
            self.y = np.reshape(self.y, (len(self.y), len(self.sequences[0]) - self.step, len(self.vocab)))

            """
        else:
            for s in self.sequences:
                for i in range(0, len(s) - self.window, self.step):
                    self.X.append([self.to_one_hot[char] for char in s[i: i + self.window]])
                    if target == 'all':
                        self.y.append([self.to_one_hot[char] for char in s[i + 1: i + self.window + 1]])
                    elif target == 'one':
                        self.y.append(s[-self.step:])

            self.X = np.reshape(self.X, (len(self.X), self.window, len(self.vocab)))
            self.y = np.reshape(self.y, (len(self.y), self.window, len(self.vocab)))

        # print("\nData shape:\nX: " + str(self.X.shape) + "\ny: " + str(self.y.shape))

    def save_generated(self, logdir, filename):
        """ Save all sequences in `self.generated` to file

        :param logdir: {str} current log directory (used for comparison sequences)
        :param filename: {str} filename to save the sequences to
        :return: saved file
        """
        with open(filename, 'w') as f:
            for s in self.generated:
                f.write(s + '\n')

        # self.ran.save_fasta(logdir + '/random_sequences.fasta')
        # self.hel.save_fasta(logdir + '/helical_sequences.fasta')

    def save_fasta(self, logdir, filename):
        """ Save all sequences in `self.generated` to fasta file

        :param logdir: {str} current log directory (used for comparison sequences)
        :param fasta filename: {str} filename to save the sequences to
        :return: saved file
        """
        with open(filename + ".fasta", "w") as output_handle:
            for i, seq in enumerate(self.generated):
                sequence = SeqRecord(Seq(seq, generic_protein), id='Sample' + str(i+1), description="Sampled from LM. EC: 4.1.1")
                SeqIO.write(sequence, output_handle, "fasta")


class Model(object):
    """
    Class containing the LSTM model to learn sequential data
    """

    def __init__(self, n_vocab, outshape, session_name, cell="LSTM", n_units=256, batch=64, layers=2, lr=0.001,
                 dropoutfract=0.1, loss='categorical_crossentropy', l2_reg=None, ask=True, seed=42):
        """ Initialize the model

        :param n_vocab: {int} length of vocabulary
        :param outshape: {int} output dimensionality of the model
        :param session_name: {str} custom name for the current session. Will create directory with this name to save
        results / logs to.
        :param n_units: {int} number of LSTM units per layer
        :param batch: {int} batch size
        :param layers: {int} number of layers in the network
        :param loss: {str} applied loss function, choose from available keras loss functions
        :param lr: {float} learning rate to use with Adam optimizer
        :param dropoutfract: {float} fraction of dropout to add to each layer. Layer1 gets 1 * value, Layer2 2 *
        value and so on.
        :param l2_reg: {float} l2 regularization for kernel
        :param seed {int} random seed used to initialize weights
        """
        random.seed(seed)
        self.seed = seed
        self.dropout = dropoutfract
        self.inshape = (None, n_vocab)
        self.outshape = outshape
        self.neurons = n_units
        self.layers = layers
        self.losses = list()
        self.val_losses = list()
        self.batchsize = batch
        self.lr = lr
        self.cv_loss = None
        self.cv_loss_std = None
        self.cv_val_loss = None
        self.cv_val_loss_std = None
        self.model = None
        self.cell = cell
        self.losstype = loss
        self.session_name = session_name
        self.logdir = './' + session_name
        self.l2 = l2_reg
        if ask and os.path.exists(self.logdir):
            decision = input('\nSession folder already exists!\n'
                             'Do you want to overwrite the previous session? [y/n] ')
            if decision in ['n', 'no', 'N', 'NO', 'No']:
                self.logdir = './' + input('Enter new session name: ')
                os.makedirs(self.logdir)
        self.checkpointdir = self.logdir + '/checkpoint/'
        if not os.path.exists(self.checkpointdir):
            os.makedirs(self.checkpointdir)
        _, _, self.vocab = _onehotencode('A')

        self.initialize_model(seed=self.seed)

    def initialize_model(self, seed=42):
        """ Method to initialize the model with all parameters saved in the attributes. This method is used during
        initialization of the class, as well as in cross-validation to reinitialize a fresh model for every fold.

        :param seed: {int} random seed to use for weight initialization

        :return: initialized model in ``self.model``
        """
        self.losses = list()
        self.val_losses = list()
        self.cv_loss = None
        self.cv_loss_std = None
        self.cv_val_loss = None
        self.cv_val_loss_std = None
        self.model = None
        weight_init = RandomNormal(mean=0.0, stddev=0.05, seed=seed)  # weights randomly between -0.05 and 0.05
        optimizer = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        if self.l2:
            l2reg = l2(self.l2)
        else:
            l2reg = None

        self.model = Sequential()
        for l in range(self.layers):
            if self.cell == "GRU":
                self.model.add(GRU(units=self.neurons,
                                   name='GRU%i' % (l + 1),
                                   input_shape=self.inshape,
                                   return_sequences=True,
                                   kernel_initializer=weight_init,
                                   kernel_regularizer=l2reg,
                                   dropout=self.dropout * (l + 1)))
            else:
                self.model.add(CuDNNLSTM(units=self.neurons,
                                         name='LSTM%i' % (l + 1),
                                         input_shape=self.inshape,
                                         return_sequences=True,
                                         kernel_initializer=weight_init,
                                         kernel_constraint=MinMaxNorm(-2.0, 2.0),
                                         recurrent_constraint=MinMaxNorm(-2.0, 2.0)))
                """
                self.model.add(LSTM(units=self.neurons,
                                    name='LSTM%i' % (l + 1),
                                    input_shape=self.inshape,
                                    return_sequences=True,
                                    kernel_initializer=weight_init,
                                    kernel_regularizer=l2reg,
                                    dropout=self.dropout * (l + 1),
                                    recurrent_dropout=self.dropout * (l + 1)))
                """
        self.model.add(TimeDistributed(Dense(self.outshape,
                                             name='Dense',
                                             activation='softmax',
                                             kernel_regularizer=l2reg,
                                             kernel_initializer=weight_init)))
        self.model.compile(loss=self.losstype, optimizer=optimizer)
        self.model.summary()
        with open(self.checkpointdir + "model.json", 'w') as f:
            json.dump(self.model.to_json(), f)
        self.get_num_params()

    def finetuneinit(self, session_name):
        """ Method to generate a new directory for finetuning a pre-existing model on a new dataset with a new name

        :param session_name: {str} new session name for finetuning
        :return: generates all necessary session folders
        """
        self.session_name = session_name
        self.logdir = './' + session_name
        if os.path.exists(self.logdir):
            decision = input('\nSession folder already exists!\n'
                             'Do you want to overwrite the previous session? [y/n] ')
            if decision in ['n', 'no', 'N', 'NO', 'No']:
                self.logdir = './' + input('Enter new session name: ')
                os.makedirs(self.logdir)
        self.checkpointdir = self.logdir + '/checkpoint/'
        if not os.path.exists(self.checkpointdir):
            os.makedirs(self.checkpointdir)

    def train(self, x, y, epochs=100, valsplit=0.2, sample=100):
        """ Train the model on given training data.

        :param x: {array} training data
        :param y: {array} targets for training data in X
        :param epochs: {int} number of epochs to train
        :param valsplit: {float} fraction of data that should be used as validation data during training
        :param sample: {int} number of sequences to sample after every training epoch
        :return: trained model and measured losses in self.model, self.losses and self.val_losses
        """
        writer = tf.summary.FileWriter('./logs/' + self.session_name, graph=sess.graph)
        for e in range(epochs):
            print("Epoch %i" % e)
            checkpoints = [ModelCheckpoint(filepath=self.checkpointdir + 'model_epoch_%i.hdf5' % e, verbose=0)]
            train_history = self.model.fit_generator(generator=x, validation_data=y, epochs=1,
                                                     callbacks=checkpoints, use_multiprocessing=False, workers=30)
            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=train_history.history['loss'][-1])])
            writer.add_summary(loss_sum, e)

            self.losses.append(train_history.history['loss'])
            if valsplit > 0.:
                self.val_losses.append(train_history.history['val_loss'])
                val_loss_sum = tf.Summary(value=[tf.Summary.Value(tag="val_loss", simple_value=train_history.history[
                    'val_loss'][-1])])
                writer.add_summary(val_loss_sum, e)
            if sample:
                for s in self.sample(sample):  # sample sequences after every training epoch
                    print(s)
        writer.close()

    def cross_val(self, x, y, epochs=100, cv=5, plot=True):
        """ Method to perform cross-validation with the model given data X, y

        :param x: {array} training data
        :param y: {array} targets for training data in X
        :param epochs: {int} number of epochs to train
        :param cv: {int} fold
        :param plot: {bool} whether the losses should be plotted and saved to the session folder
        :return:
        """
        self.losses = list()  # clean losses if already present
        self.val_losses = list()
        kf = KFold(n_splits=cv)
        cntr = 0
        for train, test in kf.split(x):
            print("\nFold %i" % (cntr + 1))
            self.initialize_model(seed=cntr)  # reinitialize every fold, otherwise it will "remember" previous data
            train_history = self.model.fit(x[train], y[train], epochs=epochs, batch_size=self.batchsize,
                                           validation_data=(x[test], y[test]))
            self.losses.append(train_history.history['loss'])
            self.val_losses.append(train_history.history['val_loss'])
            cntr += 1
        self.cv_loss = np.mean(self.losses, axis=0)
        self.cv_loss_std = np.std(self.losses, axis=0)
        self.cv_val_loss = np.mean(self.val_losses, axis=0)
        self.cv_val_loss_std = np.std(self.val_losses, axis=0)
        if plot:
            self.plot_losses(cv=True)

        # get best epoch with corresponding val_loss
        minloss = np.min(self.cv_val_loss)
        e = np.where(minloss == self.cv_val_loss)[0][0]
        print("\n%i-fold cross-validation result:\n\nBest epoch:\t%i\nVal_loss:\t%.4f" % (cv, e, minloss))
        with open(self.logdir + '/' + self.session_name + '_best_epoch.txt', 'w') as f:
            f.write("%i-fold cross-validation result:\n\nBest epoch:\t%i\nVal_loss:\t%.4f" % (cv, e, minloss))

    def plot_losses(self, show=False, cv=False):
        """Plot the losses obtained in training.

        :param show: {bool} Whether the plot should be shown or saved. If ``False``, the plot is saved to the
        session folder.
        :param cv: {bool} Whether the losses from cross-validation should be plotted. The standard deviation will be
        depicted as filled areas around the mean curve.
        :return: plot (saved) or shown interactive
        """
        fig, ax = plt.subplots()
        ax.set_title('LSTM Categorical Crossentropy Loss Plot', fontweight='bold', fontsize=16)
        if cv:
            filename = self.logdir + '/' + self.session_name + '_cv_loss_plot.pdf'
            x = range(1, len(self.cv_loss) + 1)
            ax.plot(x, self.cv_loss, '-', color='#FE4365', label='Training')
            ax.plot(x, self.cv_val_loss, '-', color='k', label='Validation')
            ax.fill_between(x, self.cv_loss + self.cv_loss_std, self.cv_loss - self.cv_loss_std,
                            facecolors='#FE4365', alpha=0.5)
            ax.fill_between(x, self.cv_val_loss + self.cv_val_loss_std, self.cv_val_loss - self.cv_val_loss_std,
                            facecolors='k', alpha=0.5)
            ax.set_xlim([0.5, len(self.cv_loss) + 0.5])
            minloss = np.min(self.cv_val_loss)
            plt.text(x=0.5, y=0.5, s='best epoch: ' + str(np.where(minloss == self.cv_val_loss)[0][0]) + ', val_loss: ' + str(minloss.round(4)),
                     transform=ax.transAxes)
        else:
            filename = self.logdir + '/' + self.session_name + '_loss_plot.pdf'
            x = range(1, len(self.losses) + 1)
            ax.plot(x, self.losses, '-', color='#FE4365', label='Training')
            if self.val_losses:
                ax.plot(x, self.val_losses, '-', color='k', label='Validation')
            ax.set_xlim([0.5, len(self.losses) + 0.5])
        ax.set_ylabel('Loss', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.legend(loc='best')
        if show:
            plt.show()
        else:
            plt.savefig(filename)

    def sample(self, num=100, minlen=7, maxlen=300, start=None, temp=2.5, show=False):
        """Invoke generation of sequence patterns through sampling from the trained model.

        :param num: {int} number of sequences to sample
        :param minlen {int} minimal allowed sequence length
        :param maxlen: {int} maximal length of each pattern generated, if 0, a random length is chosen between 7 and 50
        :param start: {str} start AA to be used for sampling. If ``None``, a random AA is chosen
        :param temp: {float} temperature value to sample at.
        :param show: {bool} whether the sampled sequences should be printed out
        :return: {array} matrix of patterns of shape (num, seqlen, inputshape[0])
        """
        print("\nSampling...\n")
        sampled = []
        lcntr = 0
        pbar = ProgressBar()
        for rs in pbar(range(num)):
            random.seed(rs)
            if not maxlen:  # if the length should be randomly sampled
                longest = np.random.randint(7, 100)
            else:
                longest = maxlen

            if start:
                start_aa = start
            else:  # generate random starting letter
                start_aa = 'B'
            sequence = start_aa  # start with starting letter

            while sequence[-1] != '-' and len(sequence) <= longest:  # sample until padding or maxlen is reached
                x, _, _ = _onehotencode(sequence)
                x = x.reshape(1, *x.shape)
                preds = self.model.predict(x)[0][-1]
                next_aa = _sample_with_temp(preds, temp=temp)
                sequence += self.vocab[next_aa]

            if start_aa == 'B':
                sequence = sequence[1:].rstrip()
            else:  # keep starting AA if chosen for sampling
                sequence = sequence.rstrip()

            if len(sequence) < minlen:  # don't take sequences shorter than the minimal length
                lcntr += 1
                continue

            sampled.append(sequence)
            if show:
                print(sequence)

        print("\t%i sequences were shorter than %i" % (lcntr, minlen))
        return sampled

    def get_num_params(self):
        """Method to get the amount of trainable parameters in the model.
        """
        trainable = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
        non_trainable = int(np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))
        print('\nMODEL PARAMETERS')
        print('Total parameters:         %i' % (trainable + non_trainable))
        print('Trainable parameters:     %i' % trainable)
        print('Non-trainable parameters: %i' % non_trainable)

    def load_model(self, filename):
        """Method to load a trained model from a hdf5 file

        :return: model loaded from file in ``self.model``
        """
        self.model.load_weights(filename)

    @staticmethod
    def reverse(inputs, axes=1):
        return K.reverse(inputs, axes=axes)


def main(infile, sessname, neurons=64, layers=2, epochs=100, batchsize=128, window=0, step=1, target='all',
         valsplit=0.2, sample=100, aa='B', temperature=2.5, cell="LSTM", dropout=0.1, train=True, learningrate=0.01,
         modfile=None, samplelength=36, pad=0, l2_rate=None, cv=None, finetune=False, references=True):
    # loading sequence data, analyze, pad and encode it
    data = SequenceHandler(window=window, step=step, refs=references)

    print("Loading sequences...")
    # data.load_sequences(infile)
    data.load_fasta(infile)

    # pad sequences
    print("\nPadding sequences...")
    data.pad_sequences(padlen=pad)

    # one-hot encode padded sequences
    print("One-hot encoding sequences...")
    data.one_hot_encode(target=target)

    if train:
        # building the LSTM model
        print("\nBuilding model...")
        model = Model(n_vocab=len(data.vocab), outshape=len(data.vocab), session_name=sessname, n_units=neurons,
                      batch=batchsize, layers=layers, cell=cell, loss='categorical_crossentropy', lr=learningrate,
                      dropoutfract=dropout, l2_reg=l2_rate, ask=True, seed=42)
        print("Model built!")

        if cv:
            print("\nPERFORMING %i-FOLD CROSS-VALIDATION...\n" % cv)
            model.cross_val(data.X, data.y, epochs=epochs, cv=cv)
            model.initialize_model(seed=42)
            model.train(data.X, data.y, epochs=epochs, valsplit=0.0, sample=0)
            model.plot_losses()
        else:
            # training model on data
            print("\nTRAINING MODEL FOR %i EPOCHS...\n" % epochs)
            model.train(data.X, data.y, epochs=epochs, valsplit=valsplit, sample=0)
            model.plot_losses()  # plot loss

        save_model_instance(model)

    elif finetune:
        print("\nUSING PRETRAINED MODEL FOR FINETUNING... (%s)\n" % modfile)
        print("Loading model...")
        model = load_model_instance(modfile)
        model.load_model(modfile)
        model.finetuneinit(sessname)  # generate new session folders for finetuning run
        print("Finetuning model...")
        model.train(data.X, data.y, epochs=epochs, valsplit=valsplit, sample=0)
        model.plot_losses()  # plot loss
        save_model_instance(model)
    else:
        print("\nUSING PRETRAINED MODEL... (%s)\n" % modfile)
        model = load_model_instance(modfile)
        model.load_model(modfile)

    print(model.get_num_params())  # print number of parameters in the model

    # generating new data through sampling
    print("\nSAMPLING %i SEQUENCES...\n" % sample)
    data.generated = model.sample(sample, start=aa, maxlen=samplelength, show=False, temp=temperature)
    # data.save_generated(model.logdir, model.logdir + '/sampled_sequences_temp' + str(temperature) + '.csv')
    data.save_fasta(model.logdir, model.logdir + '/sampled_sequences_temp' + str(temperature) + '.csv')

if __name__ == "__main__":
    # run main code
    main(infile=FLAGS.dataset, sessname=FLAGS.name, batchsize=FLAGS.batch_size, epochs=FLAGS.epochs,
         layers=FLAGS.layers, valsplit=FLAGS.valsplit, neurons=FLAGS.neurons, cell=FLAGS.cell, sample=FLAGS.sample,
         temperature=FLAGS.temp, dropout=FLAGS.dropout, train=FLAGS.train, modfile=FLAGS.modfile,
         learningrate=FLAGS.lr, cv=FLAGS.cv, samplelength=FLAGS.maxlen, window=FLAGS.window,
         step=FLAGS.step, aa=FLAGS.startchar, l2_rate=FLAGS.l2, target=FLAGS.target, pad=FLAGS.padlen,
         finetune=FLAGS.finetune, references=FLAGS.refs)

    # save used flags to log file
    _save_flags("./" + FLAGS.name + "/flags.txt")
