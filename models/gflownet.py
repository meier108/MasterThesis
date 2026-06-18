'''
GFlowNet Architecture (Jain et al., 2022)

Architecture:
  - GFlowNetMLP        : autoregressive policy network (Trajectory Balance objective)
  - ProxyMLP           : single proxy model (MLP, 2 hidden layers)
  - ProxyEnsemble      : 5 × ProxyMLP → UCB acquisition: μ + κ·σ
  - GFlowNetExperiment : integrates with the existing BaseExperiment framework
 
Key paper choices implemented:
  - Trajectory Balance loss (Eq. 11)             → _trajectory_balance_loss()
  - δ-uniform exploration mixture (Eq. 10)       → _sample_sequence()
  - γ=0.5 offline trajectory mixing (Algorithm 2) → _train_gflownet()
  - UCB acquisition with κ=0.1                   → ProxyEnsemble.ucb()
  - Reward exponent β=3                          → R(x) = proxy_score^β
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProxyMLP(nn.Module):
    '''Single proxy model: 2 hidden layers dim = 2048, ReLu activations.
    Input:
    - one-hot encoded sequence flattend shape = (L* vocab_size)
    
    Output:
    - scalar fitness prediction
    '''
    def __init__(self, input_dim: int, hidden_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # Output shape: (batch_size,)
    
class ProxyEnsemble(nn.Module):
    '''
    Ensemble of n ProxyMLP models.
    Provides mean, std and UCB = mean + x·std acquisition scores.
    '''
    def __init__(self, input_dim: int, n_members: int = 5, hidden_dim: int = 2048, kappa: float = 0.0):
        super().__init__()
        self.members = nn.ModuleList([ProxyMLP(input_dim, hidden_dim) for _ in range(n_members)])

        self.kappa = kappa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = torch.stack([member(x) for member in self.members], dim=0)  # Shape: (n_members, batch_size)
        mu = preds.mean(dim=0)  # Shape: (batch_size,)
        sigma = preds.std(dim=0)  # Shape: (batch_size,)
        return mu, sigma
    
    def ucb(self, x: torch.Tensor) -> torch.Tensor:
        mu, sigma = self.forward(x)
        return mu + self.kappa * sigma
    
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 1e-3, batch_size: int = 256, val_frac:float = 0.1):
        '''Train the ensemble on (x, y) with early stopping.
        Input: 
        - x: shape (n_samples, input_dim)
        - y: shape (n_samples,)
        ''' 
        X_t = torch.FloatTensor(x)
        y_t = torch.FloatTensor(y)

        n_val = max(1, int(len(X_t) * val_frac))
        perm = torch.randperm(len(X_t))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_train, _train = X_t[train_idx], y_t[train_idx]
        X_val, y_val = X_t[val_idx], y_t[val_idx]

        train_ds = torch.utils.data.TensorDataset(X_train, _train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for member in self.members:
            optimizer = torch.optim.Adam(member.parameters(), lr=lr, betas=(0.9, 0.999))
            best_val_loss = float('inf')
            best_state = None

            for epoch in range(epochs):
                member.train()
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    pred = member(bx)
                    loss = nn.MSELoss()(pred, by)
                    loss.backward()
                    optimizer.step()

                member.eval()
                with torch.no_grad():
                    val_loss = nn.MSELoss()(member(X_val), y_val).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in member.state_dict().items()}

            if best_state is not None:
                member.load_state_dict(best_state)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        '''Returns: mu, sigma as np.arrays'''
        self.eval()
        X_t = torch.FloatTensor(x)
        with torch.no_grad():
            mu, sigma = self.forward(X_t)
        return mu.numpy(), sigma.numpy()
    
    def ucb_numpy(self, X: np.ndarray) -> np.ndarray:
        '''Returns UCB acquisition scores as numpy array'''
        mu, sigma = self.predict(X)
        return mu + self.kappa * sigma
    
class GFlowNetMLP(nn.Module):
    '''
    Autoregressive policy network for GFlowNet. MLP with 2 hidden layers (dim=2048, ReLU activations).

    Generates sequences token by token. At each position t, the network receives
    the one-hot encoding of the previous tokens and outputs a logit for each vocabulary token.

    '''
    def __init__(self, vocab_size: int, seq_length: int, hidden_dim: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        input_dim = vocab_size * seq_length
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)  # Output logits for next token
        )
        self.log_z = nn.Parameter(torch.tensor(0.0))  # Learnable log partition function

    def _encode_partial(self, partial_seq: torch.Tensor) -> torch.Tensor:
        '''One-hot encode a partial sequence. '''
        batch_size, t = partial_seq.shape
        context = torch.zeros(batch_size, self.seq_length * self.vocab_size, device=partial_seq.device)
        if t > 0:
            oh = F.one_hot(partial_seq, num_classes=self.vocab_size).float()  # (B, t, V)
            context[:, :t * self.vocab_size] = oh.view(batch_size, -1)
        return context
    
    def forward(self, partial_seq: torch.Tensor) -> torch.Tensor:
        '''
        Input: partial_seq of shape (batch_size, t) with token indices (0 to vocab_size-1)
        Output: logits for next token of shape (batch_size, vocab_size)
        '''
        context = self._encode_partial(partial_seq)
        return self.net(context)
    
    def get_log_pf(self, sequences: torch.LongTensor) -> torch.Tensor:
        '''Compute log P_F(x) for a batch of complete sequences.'''
        batch_size= sequences.shape[0]
        log_pf = torch.zeros(batch_size, device=sequences.device)

        for t in range(self.seq_length):
            partial_seq = sequences[:, :t]
            logits = self.forward(partial_seq)  # Shape: (batch_size, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(1, sequences[:, t: t+1]).squeeze(1)
            log_pf = log_pf + token_log_probs  # Accumulate log probabilities
        return log_pf
    
    def sample_sequence(self, batch_size: int, temperature: float = 1.0, delta: float = 0.001):
        ''' 
        Sample batch size complete sequences.
        Uses the delta-uniform mixture policy from paper(Eq. 10) for exploration.
        
        Input:
        - batch_size: number of sequences to sample
        - temperature: softmax temperature for sampling higher => more diverse
        - delta: uniform exploration coefficient (0.001 for TFBind8 in paper)
        '''
        device = self.log_z.device
        self.eval()
        sequences = torch.zeros(batch_size, self.seq_length, dtype=torch.long, device=device)

        with torch.no_grad():
            for t in range(self.seq_length):
                partial_seq = sequences[:, :t]
                logits = self.forward(partial_seq) / temperature
                probs = F.softmax(logits, dim=-1)
                uniform = torch.ones_like(probs) / self.vocab_size
                mixed_probs = (1 - delta) * probs + delta * uniform

                tokens = torch.multinomial(mixed_probs, num_samples=1).squeeze(1)
                sequences[:, t] = tokens
        return sequences
    