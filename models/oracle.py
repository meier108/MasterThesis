import joblib
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from assets.data_ops import one_hot_encode_sequence, encode_sequence

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
    '''
    GB1 oracle model.

    The model takes one-hot encoded GB1 sequences, flattens them, and predicts a
    single binding score.

    Inputs:
    - L (int): flattened input dimension. For GB1 with 20-aa alphabet and
        sequence length 55, this is 1100.
    - token_to_idx (dict[str, int]): maps amino-acid tokens to indices.
    - seed (int): random seed used for weight initialization.

    Output:
    - torch.Tensor: predicted score tensor of shape (batch_size, 1).
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
        self.token_to_idx = token_to_idx

    def forward(self, x):
        '''
        Run a forward pass.

        Input:
        - x (torch.Tensor): tensor shaped (batch_size, seq_len, vocab_size).

        Output:
        - torch.Tensor: score predictions shaped (batch_size, 1).
        '''
        return self.net(x)
    
    def evaluate(self, sequence: str):
        '''
        Score one sequence string.

        Input:
        - sequence (str): amino-acid sequence.

        Output:
        - torch.Tensor: predicted score shaped (1, 1).
        '''
        x = one_hot_encode_sequence(encode_sequence(sequence, self.token_to_idx), num_tokens=len(self.token_to_idx))
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            return self.forward(x)
        
    def score_batch(self, sequences) -> np.ndarray:
        '''
        Score multiple sequences at once.

        Input:
        - sequences (list[str] | pandas.Series): sequence collection.

        Output:
        - np.ndarray: predictions shaped (n_sequences, 1).
        '''
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
        
    def train_epoch(self, train_loader, optimizer, criterion):
        '''
        Train the model for one epoch.

        Inputs:
        - train_loader (DataLoader): batches of (X, y).
        - optimizer (torch.optim.Optimizer): optimization algorithm.
        - criterion (callable): loss function.

        Output:
        - float: mean training loss over the epoch.
        '''
        self.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = self.forward(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def load_from_checkpoint(self, checkpoint_path):
        '''Load model weights from a saved checkpoint.'''
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
    
#####Code for training the GB1 oracle#####
import torch
from torch.utils.data import DataLoader, TensorDataset
from assets.data_ops import encode_sequence, one_hot_encode_sequence
from data.gb1_data import load_gb1_dataframe
from tqdm import tqdm

def build_oracle_dataset(train_data, token_to_idx):
    '''
    Build a TensorDataset for GB1 oracle training.

    Inputs:
    - train_data (list[tuple[str, float]]): (sequence, binding_score) pairs.
    - token_to_idx (dict[str, int]): amino-acid token mapping.

    Output:
    - TensorDataset: encoded inputs and target scores.
    '''
    X = np.stack([
        one_hot_encode_sequence(
            encode_sequence(seq, token_to_idx),
            num_tokens=len(token_to_idx)
        )
        for seq, _ in train_data
    ])
    y = np.asarray([score for _, score in train_data], dtype=np.float32)
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

def train_GB1_Oracle(epochs=100):
    '''
    Train a GB1 oracle on the full GB1 dataset and save it to models/oracle.pt.

    Input:
    - epochs (int): number of training epochs.

    Output:
    - None: writes trained state dict to disk.
    '''

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    token_to_idx = {token: idx for idx, token in enumerate(alphabet)}

    print("Loading GB1 dataset...")
    df = load_gb1_dataframe()
    
    oracle = Oracle_GB1(L=1100, token_to_idx=token_to_idx, seed=42)
    for param in oracle.parameters():
        param.requires_grad = True

    train_data = list(zip(df['sequence'].tolist(), df['binding_scores'].tolist()))

    optimizer = torch.optim.Adam(oracle.parameters(), lr=0.001)
    train_dataset = build_oracle_dataset(train_data, token_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.MSELoss()
    for epoch in tqdm(range(epochs), desc="Training GB1 Oracle"):
        loss = oracle.train_epoch(train_loader, optimizer, criterion)
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    torch.save(oracle.state_dict(), 'models/oracle.pt')


def load_GB1_oracle(model_path='models/oracle.pt', map_location='cpu', seed=42):
    '''
    Load a trained GB1 oracle from disk.

    Inputs:
    - model_path (str): path to the saved state dict (default: models/oracle.pt).
    - map_location (str | torch.device): target device for loading.
    - seed (int): model initialization seed before loading weights.

    Output:
    - Oracle_GB1: model with loaded weights set to eval mode.
    '''
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No oracle checkpoint found at: {model_path}")

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    token_to_idx = {token: idx for idx, token in enumerate(alphabet)}

    oracle = Oracle_GB1(L=1100, token_to_idx=token_to_idx, seed=seed)
    state_dict = torch.load(model_path, map_location=map_location)
    oracle.load_state_dict(state_dict)
    oracle.eval()
    return oracle


if __name__ == "__main__":
    print("train_GB1_Oracle was called.")
    oracle_path = 'models/oracle.pt'
    if os.path.exists(oracle_path):
        print("WARNING: models/oracle.pt already exists and may be overwritten.")

    confirmation = input("Type 'yes' to proceed with training: ").strip().lower()
    if confirmation == 'yes':
        train_GB1_Oracle(epochs=100)
        print("Training finished and oracle saved to models/oracle.pt")
    else:
        print("Aborted. Oracle training was not started.")
