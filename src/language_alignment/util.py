import inspect
import numpy as np
import os
import numbers
from scipy.stats import multivariate_normal


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


def onehot(idx, len):
    idx = np.array(idx)  # make sure this is an array
    z = np.array([0 for _ in range(len)])
    z[idx] = 1
    return z


def sample(transition_matrix,
           means, covs,
           start_state, n_samples,
           random_state):
    n_states, n_features, _ = covs.shape
    states = np.zeros(n_samples, dtype='int')
    emissions = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        if i == 0:
            prev_state = start_state
        else:
            prev_state = states[i - 1]
        state = random_state.choice(n_states,
                                    p=transition_matrix[:, prev_state])
        emissions[i] = random_state.multivariate_normal(means[state],
                                                        covs[state])
        states[i] = state
    return emissions, states


def make_data(T=20):
    """
    Sample data from a HMM model and compute associated CRF potentials.
    """

    random_state = np.random.RandomState(0)

    transition_matrix = np.array([[0.5, 0.1, 0.1],
                                  [0.3, 0.5, 0.1],
                                  [0.2, 0.4, 0.8]
                                  ])
    means = np.array([[0, 0],
                      [10, 0],
                      [5, -5]
                      ])
    covs = np.array([[[1, 0],
                      [0, 1]],
                     [[.2, 0],
                      [0, .3]],
                     [[2, 0],
                      [0, 1]]
                     ])
    start_state = 0

    emissions, states = sample(transition_matrix, means, covs, start_state,
                               n_samples=T, random_state=random_state)
    emission_log_likelihood = []
    for mean, cov in zip(means, covs):
        rv = multivariate_normal(mean, cov)
        emission_log_likelihood.append(rv.logpdf(emissions)[:, np.newaxis])
    emission_log_likelihood = np.concatenate(emission_log_likelihood, axis=1)
    log_transition_matrix = np.log(transition_matrix)

    # CRF potential from HMM model
    theta = emission_log_likelihood[:, :, np.newaxis] \
            + log_transition_matrix[np.newaxis, :, :]

    return states, emissions, theta
