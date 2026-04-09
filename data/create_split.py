import pandas as pd
import numpy as np
from typing import List, Tuple


def hamming_distance(seq1: str, seq2: str) -> int:
    """Calculate Hamming distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def select_seeds(df: pd.DataFrame, k: int, percentile_range: Tuple[float, float] = (60, 80),
                 max_percentile: float = 90, random_state: int = 42) -> List[str]:
    """
    Select K seed sequences from a moderate fitness band.

    Args:
        df: DataFrame with 'sequence' and 'binding_scores' columns
        k: Number of seeds to select
        percentile_range: (min, max) percentile range to sample from
        max_percentile: Reject seeds above this percentile
        random_state: Random seed

    Returns:
        List of seed sequences
    """
    np.random.seed(random_state)

    min_score = np.percentile(df['binding_scores'], percentile_range[0])
    max_score = np.percentile(df['binding_scores'], percentile_range[1])
    reject_threshold = np.percentile(df['binding_scores'], max_percentile)

    candidate_pool = df[(df['binding_scores'] >= min_score) &
                        (df['binding_scores'] <= max_score) &
                        (df['binding_scores'] <= reject_threshold)]

    seeds = candidate_pool.sample(n=min(k, len(candidate_pool)), random_state=random_state)
    return seeds['sequence'].tolist()


def assign_to_clusters(df: pd.DataFrame, seeds: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign all sequences to their nearest seed (cluster assignment).

    Args:
        df: DataFrame with 'sequence' column
        seeds: List of seed sequences

    Returns:
        Tuple of (cluster_assignments, distances_to_seed)
        - cluster_assignments: array of cluster indices for each sequence
        - distances_to_seed: array of Hamming distances to assigned seed
    """
    n_sequences = len(df)
    cluster_assignments = np.zeros(n_sequences, dtype=int)
    distances_to_seed = np.zeros(n_sequences, dtype=int)

    for idx, seq in enumerate(df['sequence']):
        # Find nearest seed
        min_dist = float('inf')
        nearest_seed_idx = 0

        for seed_idx, seed in enumerate(seeds):
            dist = hamming_distance(seq, seed)
            if dist < min_dist:
                min_dist = dist
                nearest_seed_idx = seed_idx

        cluster_assignments[idx] = nearest_seed_idx
        distances_to_seed[idx] = min_dist

    return cluster_assignments, distances_to_seed


def create_split(df: pd.DataFrame,
                 k_seeds: int = 50,
                 n_training_seeds: int = 5,
                 train_fraction: float = 0.70,
                 test_a_fraction: float = 0.10,
                 test_b_fraction: float = 0.10,
                 test_c_fraction: float = 0.10,
                 fitness_percentile_cutoff: float = 50.0,
                 training_radius_percentile: float = 70.0,
                 random_state: int = 42) -> pd.DataFrame:
    """
    Create a realistic cluster-based train/test split for TF-DNA binding data.

    Uses ALL sequences by:
    1. Selecting K seed sequences from moderate fitness band within low-fitness subset
    2. Assigning every sequence to its nearest seed (cluster)
    3. Splitting clusters into train vs test_c clusters
       - CRITICAL: Only n_training_seeds clusters used for training (realistic: 3-5)
       - Remaining (k_seeds - n_training_seeds) clusters go to test_c
    4. Within train clusters:
       - Define training radius (e.g., 70th percentile of distances)
       - Within radius: randomly split into train (low-fitness only) and test_a (any fitness)
       - Beyond radius: test_b (any fitness)
    5. CRITICALLY: Only train set is restricted to low-fitness (bottom X%)
    6. Test sets can contain ANY fitness level

    This mimics offline model-based optimization where:
    - Training: only observe poor-to-moderate designs (bottom X% fitness) within training radius
      from a SMALL number of known binders (3-5 seeds), reflecting realistic experimental scenarios
    - Test A: holdout from SAME distance range (tests interpolation)
    - Test B: sequences BEYOND training radius (tests extrapolation)
    - Test C: different clusters entirely (tests generalization to novel motifs)

    Args:
        df: DataFrame with 'sequence' and 'binding_scores' columns
        k_seeds: Total number of seed sequences/clusters to discover in data (e.g., 50-100)
        n_training_seeds: Number of seeds used for training (realistic: 3-5, mimics starting
                         with a small number of known binders)
        train_fraction: Fraction of within-radius data for training (vs test_a)
        test_a_fraction: Fraction of within-radius data for test_a holdout
        test_b_fraction: Fraction of beyond-radius data (informational, uses all beyond radius)
        test_c_fraction: Fraction for test C (novel-seed shift, different clusters)
        fitness_percentile_cutoff: Only train uses sequences below this percentile (default 50)
        training_radius_percentile: Percentile of distances to use as training radius (default 70)
        random_state: Random seed

    Returns:
        DataFrame with 'split' column added ('train', 'test_a', 'test_b', 'test_c', 'unused')
        Also adds 'cluster' and 'distance_to_seed' columns for analysis
    """
    np.random.seed(random_state)
    df = df.copy()
    df['split'] = 'unused'

    # Step 1: Calculate fitness threshold for training restriction
    fitness_threshold = np.percentile(df['binding_scores'], fitness_percentile_cutoff)
    low_fitness_mask = df['binding_scores'] <= fitness_threshold

    # Work with low-fitness subset to select seeds
    df_low_fitness = df[low_fitness_mask].copy()

    # Step 2: Select seeds from moderate fitness band (within low-fitness subset)
    seeds = select_seeds(df_low_fitness, k_seeds, random_state=random_state)

    # Step 3: Assign ALL sequences to their nearest seed
    cluster_assignments, distances_to_seed = assign_to_clusters(df, seeds)
    df['cluster'] = cluster_assignments
    df['distance_to_seed'] = distances_to_seed

    # Step 4: Split seeds into train clusters and test_c clusters
    # CRITICAL: Use only n_training_seeds for training (realistic: 3-5 known binders)
    # All remaining clusters (k_seeds - n_training_seeds) go to test_c (novel motifs)
    assert n_training_seeds < k_seeds, f"n_training_seeds ({n_training_seeds}) must be < k_seeds ({k_seeds})"

    train_seed_indices = np.random.choice(k_seeds, size=n_training_seeds, replace=False)
    test_c_seed_indices = [i for i in range(k_seeds) if i not in train_seed_indices]

    # Step 5: Assign test_c sequences (ALL sequences in test_c clusters, any fitness)
    test_c_mask = df['cluster'].isin(test_c_seed_indices)
    df.loc[test_c_mask, 'split'] = 'test_c'

    # Step 6: Split remaining sequences (from train clusters) into train/test_a/test_b
    train_cluster_mask = df['cluster'].isin(train_seed_indices)
    all_train_cluster_df = df[train_cluster_mask].copy()

    # Calculate training radius (e.g., 70th percentile of distances in train clusters)
    training_radius = np.percentile(all_train_cluster_df['distance_to_seed'], training_radius_percentile)

    # Split by distance relative to training radius
    within_radius_mask = all_train_cluster_df['distance_to_seed'] <= training_radius
    beyond_radius_mask = all_train_cluster_df['distance_to_seed'] > training_radius

    within_radius_df = all_train_cluster_df[within_radius_mask].copy()
    beyond_radius_df = all_train_cluster_df[beyond_radius_mask].copy()

    # For sequences WITHIN training radius: randomly split into train and test_a
    # Both come from same distance distribution, but train is low-fitness only
    n_within = len(within_radius_df)
    n_train_target = int(n_within * (train_fraction / (train_fraction + test_a_fraction)))

    # Shuffle within-radius sequences for random split
    within_radius_shuffled = within_radius_df.sample(frac=1.0, random_state=random_state)

    # Split into train candidates and test_a candidates
    train_candidates = within_radius_shuffled.iloc[:n_train_target]
    test_a_candidates = within_radius_shuffled.iloc[n_train_target:]

    # For TRAIN: only use low-fitness sequences (offline MBO constraint)
    train_low_fitness = train_candidates[train_candidates['binding_scores'] <= fitness_threshold]
    df.loc[train_low_fitness.index, 'split'] = 'train'

    # High-fitness sequences in train_candidates go to test_a (solves "unused" problem!)
    train_high_fitness = train_candidates[train_candidates['binding_scores'] > fitness_threshold]
    df.loc[train_high_fitness.index, 'split'] = 'test_a'

    # For TEST A: all remaining test_a candidates (any fitness level)
    df.loc[test_a_candidates.index, 'split'] = 'test_a'

    # For TEST B: all sequences BEYOND training radius (any fitness level)
    df.loc[beyond_radius_df.index, 'split'] = 'test_b'

    return df
