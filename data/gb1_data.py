"""GB1 dataset helpers and a small CLI for generating mutated sequences.

The module serves two purposes:
- provide reusable functions for notebooks and other Python code
- offer a command line entry point for dataset generation
"""

from __future__ import annotations
from tqdm.auto import tqdm
import time



import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import mavenn

import pandas as pd
import numpy as np

from numba import njit, prange
from models.single_mutant_walker import SingleMutantWalker
import data.create_split as create_split



def load_gb1_data():
    """Load the GB1 dataset from the mavenn package."""
    data = mavenn.load_example_dataset("gb1")
    return data

def select_columns(data):
    """Select the relevant columns from the GB1 dataset."""
    return data[['x', 'y']]

def load_gb1_dataframe():
    """Load GB1 and return a DataFrame with sequence, score, and split columns."""
    data = load_gb1_data()
    df = pd.DataFrame(
        {
            "sequence": data["x"],
            "binding_scores": data["y"],
            "split": ["None"] * len(data["x"]),
        }
    )
    df = create_split.create_split(df)
    return df

# Compute the minimum Hamming distance from each test sequence to the train set.
@njit(parallel=True)
def min_hamming_distance(X_test, X_train):
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

# Assign qualitative labels based on the Hamming distance bins below.
def assign_labels(min_dists):
    bins = [0, 3, 7, 10, 15, 20, np.inf]
    labels = ['near', 'close', 'medium', 'far', 'very_far', 'distant']
    indices = np.digitize(min_dists, bins)
    named_labels = np.array(labels)[indices - 1]
    print(f"Label distribution: {pd.Series(named_labels).value_counts()}")
    return named_labels


def create_dataset(size=1000, file_path=None, seed=42):
    """Create a GB1 dataset by mutating train sequences.

    The generated sequences are compared against the GB1 train split and their
    minimum Hamming distance is stored alongside each sequence.

    Args:
        size: Number of sequences to generate.
        file_path: Optional CSV path for saving the generated data.
        seed: Random seed for reproducibility.

    Returns:
        A pandas DataFrame containing the generated dataset.
    """

    np.random.seed(seed)

    # Load the GB1 dataset and select the training split
    df = load_gb1_dataframe()
    df_train = df[df['split'] == 'train']
    train_sequences = df_train['sequence'].astype(str).tolist()
    train_set = set(train_sequences)
    train_array = np.array([list(seq) for seq in train_sequences], dtype='U1')
    print(f"Number of training samples: {len(df_train)}")

    # Set up the SingleMutantWalker
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    seq_length = len(df_train['sequence'].iloc[0])
    smw = SingleMutantWalker(alphabet, seq_length)

    # Try a range of mutation counts so the generated sequences span multiple
    # similarity levels relative to the training set.
    n_mutations_list = np.array([1, 3, 5, 6, 8, 10, 20, 40])
    generated_sequences = {
        'sequence': [],
        'min_hamming_distance': [],
    }

    def select_random_sequence(train_sequences):
        index = np.random.randint(len(train_sequences))
        return train_sequences[index]

    def mutate_sequence(smw, n_mutations, train_set, train_sequences):
        sequences = []
        seed = select_random_sequence(train_sequences)
        for _ in range(n_mutations):
            seed = smw.mutate_sequence(seed)
            if seed not in train_set:
                sequences.append(seed)
        return sequences
    
    def encode_sequence(sequence):
        return np.array(list(sequence), dtype='U1')

    def generate_sequences(smw, n_mutations_list, dataset_size, train_set, train_sequences, train_array):
        # tqdm keeps long runs readable while the generator fills the target size.
        t = tqdm(total=dataset_size, desc="Generating Sequences")

        while len(generated_sequences['sequence']) < dataset_size:
            n_mutations = np.random.choice(n_mutations_list)
            mutated_seqs = mutate_sequence(smw, n_mutations, train_set, train_sequences)
            for seq in mutated_seqs:
                generated_sequences['sequence'].append(seq)
                generated_sequences['min_hamming_distance'].append(min_hamming_distance(encode_sequence(seq)[None, :], train_array)[0])
            time.sleep(0.1)  # Sleep to allow tqdm to update
            t.update(len(generated_sequences['sequence']) - t.n)  # Update tqdm with the number of new sequences generated
        print(f"Generated {len(generated_sequences['sequence'])} sequences")
        
    generate_sequences(smw, n_mutations_list, size, train_set, train_sequences, train_array)
    df_generated = pd.DataFrame(generated_sequences)

    if file_path is not None:
        df_generated.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")

    return df_generated


def build_arg_parser():
    """Create the command line parser for GB1 dataset generation."""

    parser = argparse.ArgumentParser(
        description="Generate a GB1 mutation dataset and optionally save it as CSV.",
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of generated sequences to create.",
    )
    parser.add_argument(
        "--path",
        dest="file_path",
        required=True,
        help="CSV path used to save the generated dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sequence generation.",
    )
    return parser


def main():
    """Parse CLI arguments and generate the requested GB1 dataset."""

    parser = build_arg_parser()
    args = parser.parse_args()

    print("Generate a GB1 mutation dataset by sampling train sequences and mutating them.")
    print(f"Requested size: {args.size}")
    print(f"Output path: {args.file_path}")
    print(f"Seed: {args.seed}")

    create_dataset(size=args.size, file_path=args.file_path, seed=args.seed)
    

if __name__ == "__main__":
    main()
