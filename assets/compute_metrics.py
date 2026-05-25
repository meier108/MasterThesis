import pandas as pd
import numpy as np

from numba import prange, njit

@njit(parallel=True)
def min_hamming_distance(X_test, X_train):
    '''Compute the minimum Hamming distance from each sequence in X_test to any sequence in X_train.
    Input:
        X_test: A 2D numpy array of shape (n_test, seq_len) containing the test sequences.
        X_train: A 2D numpy array of shape (n_train, seq_len) containing the training sequences.
    Output:
        A 1D numpy array of shape (n_test,) containing the minimum Hamming distance'''
    n_seqs = X_test.shape[0]
    n_train = X_train.shape[0]
    seq_len = X_test.shape[1]
    min_dists = np.empty(n_seqs, dtype=np.int16)

    for i in prange(n_seqs):
        best = seq_len + 1
        for j in range(n_train):
            d = 0
            for k in range(seq_len):
                if X_test[i, k] != X_train[j, k]:
                    d += 1
                    if d >= best:
                        break
            if d < best:
                best = d
        min_dists[i] = best
    return min_dists

def assign_labels(min_dists):
    '''Assign labels to sequences based on their minimum Hamming distance.
    Input:
        min_dists: A 1D numpy array of shape (n_test,) containing the minimum Hamming distances.
    Output:
        A 1D numpy array of shape (n_test,) containing the assigned labels.'''
    bins = [0, 3, 7, 10, 15, 20, np.inf]
    labels = ['near', 'close', 'medium', 'far', 'very_far', 'distant']
    indices = np.digitize(min_dists, bins)
    named_labels = np.array(labels)[indices - 1]
    print(f"Label distribution: {pd.Series(named_labels).value_counts()}")
    return named_labels