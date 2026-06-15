"""Metrics computation utilities."""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from numba import prange, njit


@njit(parallel=True)
def min_hamming_distance(X_test, X_train):
    '''Compute the minimum Hamming distance from each sequence in X_test to any sequence in X_train.
    
    Input:
        X_test: A 2D numpy array of shape (n_test, seq_len) containing the test sequences.
        X_train: A 2D numpy array of shape (n_train, seq_len) containing the training sequences.
    
    Output:
        A 1D numpy array of shape (n_test,) containing the minimum Hamming distance
    '''
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

# Mean pairwise hamming distance within the 100 mutants per iteration
def hamming_distance(seq1, seq2):
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))

def mean_pairwise_hamming_distance(df):
    sequences = df['sequence'].tolist()
    n = len(sequences)
    if n < 2:
        return 0.0
    total_distance = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += hamming_distance(sequences[i], sequences[j])
            count += 1
    return total_distance / count

def assign_labels(min_dists):
    '''Assign labels to sequences based on their minimum Hamming distance.
    
    Input:
        min_dists: A 1D numpy array of shape (n_test,) containing the minimum Hamming distances.
    
    Output:
        A 1D numpy array of shape (n_test,) containing the assigned labels.
    '''
    bins = [0, 3, 7, 10, 15, 20, np.inf]
    labels = ['near', 'close', 'medium', 'far', 'very_far', 'distant']
    indices = np.digitize(min_dists, bins)
    named_labels = np.array(labels)[indices - 1]
    print(f"Label distribution: {pd.Series(named_labels).value_counts()}")
    return named_labels


def compute_model_metrics(df, target_col='oracle_score', rf_pred_col='rf_prediction', mlp_pred_col='mlp_prediction'):
    '''Compute and display metrics for Random Forest and MLP models.
    
    Calculates Spearman correlation, MSE, bias, and variance for both models.
    
    Input:
        df: DataFrame containing the oracle scores and model predictions
        target_col: Column name for the oracle/true scores (default: 'oracle_score')
        rf_pred_col: Column name for RF predictions (default: 'rf_prediction')
        mlp_pred_col: Column name for MLP predictions (default: 'mlp_prediction')
    
    Output:
        Prints a formatted table with metrics for both models
    '''
    
    # Random Forest metrics
    spearmanr_rf = spearmanr(df[target_col], df[rf_pred_col])
    mse_rf = mean_squared_error(df[target_col], df[rf_pred_col])
    bias_rf = (df[target_col] - df[rf_pred_col]).mean()
    variance_rf = ((df[rf_pred_col] - df[rf_pred_col].mean()) ** 2).mean()
    
    # MLP metrics
    spearmanr_mlp = spearmanr(df[target_col], df[mlp_pred_col])
    mse_mlp = mean_squared_error(df[target_col], df[mlp_pred_col])
    bias_mlp = (df[target_col] - df[mlp_pred_col]).mean()
    variance_mlp = ((df[mlp_pred_col] - df[mlp_pred_col].mean()) ** 2).mean()
    
    # Create results dataframe for nice table display
    results = pd.DataFrame({
        'Model': ['Random Forest', 'MLP'],
        'Spearman ρ': [spearmanr_rf.correlation, spearmanr_mlp.correlation],
        'MSE': [mse_rf, mse_mlp],
        'Bias': [bias_rf, bias_mlp],
        'Variance': [variance_rf, variance_mlp]
    })
    
    # Format and print the table
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    for col in ['Spearman ρ', 'MSE', 'Bias', 'Variance']:
        results[col] = results[col].apply(lambda x: f'{x:.4f}')
    print(results.to_string(index=False))
    print("="*80 + "\n")
