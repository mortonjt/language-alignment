from language_alignment.language_model import Elmo, OneHot, Roberta, Bert, Unirep
import os

_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_model(path):
    return os.path.join(_ROOT, 'models', path)


pretrained_language_models = {
    'elmo': (Elmo, get_model('lstm_lm.hdf5')),
    'onehot': (OneHot, None),
    'roberta': (Roberta, None),
    'bert': (Bert, None),
    'unirep': (Unirep, None)
}
