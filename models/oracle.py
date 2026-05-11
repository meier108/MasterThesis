import joblib
import pandas as pd
import torch
import torch.nn as nn
from assets.data_ops import encode_sequence, one_hot_encode_sequence
import numpy as np


class Oracle_TFBind8:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._scores = dict(zip(data["sequence"], data["binding_scores"]))
        self._splits = dict(zip(data["sequence"], data["split"]))

    def exists(self, sequence: str) -> bool:
        """Check if the given sequence exists in the dataset."""
        return sequence in self._scores

    def get_score(self, sequence: str):
        """Return the binding score for the given sequence, or None if missing."""
        return self._scores.get(sequence)

    def evaluate(self, sequence: str):
        """Compatibility wrapper used by the optimization loop."""
        return self.get_score(sequence)

    def get_split(self, sequence: str):
        """Return the split label for a sequence, or None if unknown."""
        return self._splits.get(sequence)
    
    def get_df(self):
        """Return the underlying DataFrame."""
        return self.data

class Oracle_GB1(nn.Module):
    '''GB1 oracle class. The GB1 oracle is a ramdom MLP traind on the whole GB1 dataset.
    Following Brooks & Listgarten 2023:
    '''
    def __init__(self, L, token_to_idx, seed):
        '''Input ( L, 4) -> Flatten(L*4) -> Dense(50) -> Dense(50) -> Dense(1).
        Weights init with Glorot Uniform, never trained. '''
        super().__init__()
        torch.manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(L, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1)
        )

        self.__init_glorot_uniform()
        for param in self.parameters():
            param.requires_grad = False

        self.token_to_idx = token_to_idx

    def __init_glorot_uniform(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
    def evaluate(self, sequence: str):
        '''Compatibility wrapper used by the optimization loop.'''
        x = one_hot_encode_sequence(encode_sequence(sequence, self.token_to_idx), num_tokens=len(self.token_to_idx))
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            return self.forward(x)
        
    def score_batch(self, sequences) -> np.ndarray:
        '''Score a list or pandas Series of sequences. Used for bulk scoring.'''
        X = np.stack([
            one_hot_encode_sequence(
                encode_sequence(seq, self.token_to_idx),
                num_tokens=len(self.token_to_idx)
            )
            for seq in sequences
        ])
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X).numpy()