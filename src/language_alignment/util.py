import inspect
import os


def get_data_path(fn, subfolder='data'):
    """Return path to filename ``fn`` in the data folder.
    During testing it is often necessary to load data files. This
    function returns the full path to files in the ``data`` subfolder
    by default.
    Parameters
    ----------
    fn : str
        File name.
    subfolder : str, defaults to ``data``
        Name of the subfolder that contains the data.
    Returns
    -------
    str
        Inferred absolute path to the test data for the module where
        ``get_data_path(fn)`` is called.
    Notes
    -----
    The requested path may not point to an existing file, as its
    existence is not checked.
    This is from skbio's code base
    https://github.com/biocore/scikit-bio/blob/master/skbio/util/_testing.py#L50
    """
    # getouterframes returns a list of tuples: the second tuple
    # contains info about the caller, and the second element is its
    # filename
    callers_filename = inspect.getouterframes(inspect.currentframe())[1][1]
    path = os.path.dirname(os.path.abspath(callers_filename))
    data_path = os.path.join(path, subfolder, fn)
    return data_path


def check_random_state(seed):
    """ Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    Note
    ----
    This is from sklearn
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
