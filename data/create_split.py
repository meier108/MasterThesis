import pandas as pd
import numpy as np


def create_split(
    df: pd.DataFrame,
    k_seeds: int = 50,
    n_training_seeds: int = 10,
    fitness_percentile_cutoff: float = 50.0,
    training_radius_percentile: float = 70.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assign a 'split' column with values 'train' or 'unused'.

    Training sequences are selected by:
    1. Restricting to the bottom `fitness_percentile_cutoff` % by fitness score.
    2. Selecting `k_seeds` seed sequences from the moderate-fitness band within
       that low-fitness subset, then keeping only `n_training_seeds` of those clusters.
    3. Within the training clusters, only sequences within the `training_radius_percentile`
       percentile of Hamming distance to their seed are included.

    Args:
        df:                        DataFrame with 'sequence' and 'binding_scores' columns.
        k_seeds:                   Total number of seed clusters to discover.
        n_training_seeds:          Number of those clusters used for training.
        fitness_percentile_cutoff: Bottom-X% fitness threshold for training (default 50).
        training_radius_percentile: Hamming-distance radius percentile within clusters (default 70).
        random_state:              Random seed.

    Returns:
        DataFrame with 'split' column ('train' or 'unused').
    """
    np.random.seed(random_state)
    df = df.copy()
    df['split'] = 'unused'

    fitness_threshold = np.percentile(df['binding_scores'], fitness_percentile_cutoff)
    low_fitness_mask = df['binding_scores'] <= fitness_threshold
    df_low = df[low_fitness_mask]

    # Select seeds from the moderate-fitness band within the low-fitness subset
    lo = np.percentile(df_low['binding_scores'], 60)
    hi = np.percentile(df_low['binding_scores'], 80)
    candidate_pool = df_low[(df_low['binding_scores'] >= lo) & (df_low['binding_scores'] <= hi)]
    seeds = (
        candidate_pool
        .sample(n=min(k_seeds, len(candidate_pool)), random_state=random_state)['sequence']
        .tolist()
    )

    # Vectorised Hamming distance: (n_sequences, n_seeds)
    seqs_arr  = np.frompyfunc(list, 1, 1)(df['sequence'].values)
    seqs_mat  = np.array(seqs_arr.tolist())                        # (N, L)
    seeds_arr = np.frompyfunc(list, 1, 1)(np.array(seeds))
    seeds_mat = np.array(seeds_arr.tolist())                       # (K, L)

    dist_matrix = (seqs_mat[:, None, :] != seeds_mat[None, :, :]).sum(axis=2)  # (N, K)
    cluster_assignments = dist_matrix.argmin(axis=1)
    distances_to_seed   = dist_matrix.min(axis=1)

    df['_cluster'] = cluster_assignments
    df['_dist']    = distances_to_seed

    # Choose which seed clusters are used for training
    train_seed_idx = np.random.choice(len(seeds), size=n_training_seeds, replace=False)
    in_train_cluster = df['_cluster'].isin(train_seed_idx)

    # Training radius based on distances within the selected clusters
    training_radius = np.percentile(
        df.loc[in_train_cluster, '_dist'], training_radius_percentile
    )

    train_mask = (
        in_train_cluster
        & (df['_dist'] <= training_radius)
        & low_fitness_mask
    )
    df.loc[train_mask, 'split'] = 'train'

    df.drop(columns=['_cluster', '_dist'], inplace=True)

    n_train = train_mask.sum()
    max_score = df.loc[train_mask, 'binding_scores'].max()
    min_score = df.loc[train_mask, 'binding_scores'].min()
    print(f"Selected {n_train} training samples.")
    print(f"Training score range: {min_score} - {max_score}")

    return df
