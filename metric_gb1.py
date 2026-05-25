import numpy as np
import pandas as pd


def create_dataframe(smw, oracle, num_samples, train_data, random_state = 42):
    data = {
        'sequence': None,
        'score': None,
        'ham_distance': None,
    }

    # Select random sample from the trainig data
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(train_data), size=1000, replace=False)
    seed_sequence = train_data.iloc[sample_indices]['sequence'].values

    for _ in range(num_samples):
        mutations = smw.mutate(seed_sequence, num_mutations=1)
        for mutated_sequence in mutations:
            if mutated_sequence in train_data['sequence'].values:
                continue

            score = oracle(mutated_sequence)
            ham_distance = smw.hamming_distance(seed_sequence, mutated_sequence)
            
            data['sequence'] = mutated_sequence
            data['score'] = score
            data['ham_distance'] = ham_distance