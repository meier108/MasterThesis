"""Reinforcement Learning experiment with LSTM and RF surrogate."""

from typing import List
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from base_experiment import BaseExperiment
from trajectory import TrajectoryRecord
from experiment_config import ExperimentConfig
from models import oracle, random_forest, goal_directed_lstm, mlp
from assets.data_ops import (
    load_data, build_tfbind8_dataframe, encode_sequence, 
    one_hot_encode_sequence, decode_sequence
)


class RLExperiment(BaseExperiment):
    """Reinforcement Learning with LSTM generator and MLP surrogate."""
    
    def __init__(self, config: ExperimentConfig, run_id: int):
        super().__init__(config, run_id)
        
        # RL-specific state
        self.lstm_model = None
        self.rl_config = config.rl_config
        self.X_train = None  # For replay
        self.y_train = None  # For replay
        self.seen_sequences = set() # For diversity filtering
        # For diversity filtering during iterations
        self.all_generated_sequences = []
        self.max_score = 1.0
        
    def setup(self):
        """Initialize oracle, surrogate (MLP), pretrain LSTM."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        print(f"after set_seed:    {hash(torch.get_rng_state().numpy().tobytes())}")
        if self.config.dataset == "tfbind8":
            self._setup_tfbind8()
        elif self.config.dataset == "gb1":
            self._setup_gb1()
            print(f"after _setup_gb1:  {hash(torch.get_rng_state().numpy().tobytes())}")
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        # Train MLP surrogate
        self.surrogate = mlp.MLPModel()
       
        X_train_one_hot = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx)) 
            for seq in self.X_train
        ])
        self.surrogate.fit(X_train_one_hot, self.y_train)
        
        # Initialize and pretrain LSTM
        seq_length = len(self.train_df["sequence"].iloc[0])
        self.lstm_model = goal_directed_lstm.GoalDirectedLSTM(
            vocab_size=len(self.token_to_idx),
            embedding_dim=32,
            sequence_length=seq_length,
            hidden_dim=128,
        )
        self._pretrain_lstm()
        # Prepare training data for Hamming distance computation
        self.X_train_encoded = self.X_train.copy()
    
    def _setup_tfbind8(self):
        """Setup TFBind8 dataset."""
        alphabet = ["A", "C", "G", "T"]
        self.token_to_idx = {token: idx for idx, token in enumerate(alphabet)}
        
        x, y = load_data(name="tfbind8")
        df = build_tfbind8_dataframe(x, y, alphabet)
        self.train_df = df[df["split"] == "train"].copy()
        
        self.oracle = oracle.Oracle_TFBind8(df)
        
        # Prepare training data
        self.X_train = np.array([
            encode_sequence(seq, self.token_to_idx) 
            for seq in self.train_df["sequence"]
        ])
        self.y_train = self.train_df["binding_scores"].values
    
    def _setup_gb1(self):
        """Setup GB1 dataset."""
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.token_to_idx = {token: idx for idx, token in enumerate(alphabet)}
        print(f"rng 1: {hash(torch.get_rng_state().numpy().tobytes())}")
        df = load_data(name="gb1")
        self.train_df = df[df["split"] == "train"].copy()
        
        # Get sequence length
        one_hot_sequence = one_hot_encode_sequence(
            encode_sequence(self.train_df['sequence'].iloc[0], self.token_to_idx),
            num_tokens=len(self.token_to_idx)
        )
        L = one_hot_sequence.shape[0]
        print(f"rng 2: {hash(torch.get_rng_state().numpy().tobytes())}")
        self.oracle = oracle.load_GB1_oracle()
        
        #Oracle uses a fixed seed, this overrides rng therefore we need to reset it
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Prepare training data
        self.X_train = np.array([
            encode_sequence(seq, self.token_to_idx) 
            for seq in self.train_df["sequence"]
        ])
        print(f"rng 3: {hash(torch.get_rng_state().numpy().tobytes())}")
        # Score training sequences with oracle
        self.y_train = self.oracle.score_batch(self.train_df["sequence"])
        self.max_score = self.y_train.max()
        print(f"rng 4: {hash(torch.get_rng_state().numpy().tobytes())}")

    def _pretrain_lstm(self):
        """Pretrain LSTM on training data."""
        X_train_tensor = torch.LongTensor(self.X_train)
        y_train_tensor = torch.FloatTensor(self.y_train).view(-1, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        self._train_goal_directed_model(
            train_loader, 
            epochs=self.rl_config.lstm_epochs,
            entropy_weight=self.rl_config.entropy_weight
        )
    
    def _train_goal_directed_model(self, loader, epochs=20, lr=0.001, entropy_weight=0.01):
        """Train LSTM with entropy regularization."""
        optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.lstm_model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                
                input_seq = batch_x[:, :-1]
                target_part = batch_x[:, 1:]
                logits, _ = self.lstm_model(input_seq, batch_y)
                
                ce_loss = criterion(
                    logits.reshape(-1, self.lstm_model.vocab_size), 
                    target_part.reshape(-1)
                )
                
                # Entropy regularization
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                loss = ce_loss - entropy_weight * entropy
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f'Pretraining epoch {epoch}/{epochs}, Loss: {epoch_loss/len(loader):.4f}')
    
    def _diversity_filter(self, sequences_np, scores, real_scores, min_hamming=2):
        """Greedily select sequences with minimum Hamming distance."""
        if len(sequences_np) == 0:
            return sequences_np, scores, real_scores
        
        selected_idx = [0]
        for i in range(1, len(sequences_np)):
            is_diverse = True
            for j in selected_idx:
                dist = np.sum(sequences_np[i] != sequences_np[j])
                if dist < min_hamming:
                    is_diverse = False
                    break
            if is_diverse:
                selected_idx.append(i)
        
        idx = np.array(selected_idx)
        return sequences_np[idx], scores[idx], real_scores[idx]
    
    def _evaluate_oracle(self, sequence_encoded):
        """Evaluate oracle for a single encoded sequence. Returns a Python float."""
        if self.config.dataset == "tfbind8":
            seq_str = decode_sequence(sequence_encoded, alphabet=list(self.token_to_idx.keys()))
            return self.oracle.evaluate(seq_str)
        elif self.config.dataset == "gb1":
            # For GB1, convert to string and evaluate (oracle.evaluate() returns float)
            seq_str = decode_sequence(sequence_encoded, alphabet=list(self.token_to_idx.keys()))
            return self.oracle.evaluate(seq_str)
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
    
    def _predict_surrogate_batch(self, sequences_encoded):
        """Predict surrogate scores for a batch of encoded sequences."""
        X_one_hot = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx)) 
            for seq in sequences_encoded
        ])
        predictions = self.surrogate.predict(X_one_hot)
        return predictions
    
    def run_single_iteration(self, iteration: int) -> List[TrajectoryRecord]:
        """
        Run a single RL iteration: generate, score, select, filter.
        
        Returns records for all diverse sequences selected in this iteration.
        """
        records = []
        
        # Anneal temperature
        temp = self.rl_config.temp_start + (
            self.rl_config.temp_end - self.rl_config.temp_start
        ) * (iteration / max(self.rl_config.num_iterations - 1, 1))
        
        print(f'RL Iteration {iteration+1}/{self.rl_config.num_iterations} (temp={temp:.2f})')
        
        # Generate sequences
        new_sequences = []
        # TODO: Reset seed because of RNG problem
        #torch.manual_seed(self.seed)
        for _ in range(self.rl_config.batch_size):
            seq = self.lstm_model.generate(goal_score=self.max_score, temperature=temp) #TODO: goal:socre changed from 1.1 to 0.5
            new_sequences.append(seq)
        
        new_sequences_np = np.array(new_sequences)
        self.all_generated_sequences.extend(new_sequences_np)
        
        # Score with surrogate (MLP)
        surrogate_scores = self._predict_surrogate_batch(new_sequences_np)
        
        # Score with oracle
        oracle_scores = np.array([
            self._evaluate_oracle(seq) for seq in new_sequences_np
        ])

        self.max_score = max(self.max_score, surrogate_scores.max()) #Changed from oracle_scores.max() to surrogate scores, as we dont know oracle scores
        # Select top 20% by surrogate score
        threshold = np.percentile(surrogate_scores, 80)
        top_mask = surrogate_scores >= threshold
        top_sequences = new_sequences_np[top_mask]
        top_surrogate_scores = surrogate_scores[top_mask]
        top_oracle_scores = oracle_scores[top_mask]
        
        # Apply diversity filter
        if len(top_sequences) > 1:
            top_sequences, top_surrogate_scores, top_oracle_scores = self._diversity_filter(
                top_sequences, top_surrogate_scores, top_oracle_scores,
                min_hamming=self.rl_config.min_hamming
            )
        
        # TODO: Filter out sequences already seen in previous iterations
        
        new_mask = []
        for i, seq_encoded in enumerate(top_sequences):
            seq_str = decode_sequence(seq_encoded, alphabet=list(self.token_to_idx.keys()))
            if seq_str not in self.seen_sequences:
                new_mask.append(i)
                self.seen_sequences.add(seq_str)
        
        if new_mask:
            new_mask = np.array(new_mask)
            top_sequences = top_sequences[new_mask]
            top_surrogate_scores = top_surrogate_scores[new_mask]
            top_oracle_scores = top_oracle_scores[new_mask]
        else:
            # If all sequences were already seen, keep the best one
            top_sequences = top_sequences[:1] if len(top_sequences) > 0 else np.array([])
            top_surrogate_scores = top_surrogate_scores[:1] if len(top_surrogate_scores) > 0 else np.array([])
            top_oracle_scores = top_oracle_scores[:1] if len(top_oracle_scores) > 0 else np.array([])
        

        # Create trajectory records
        for seq_encoded, surrogate_score, oracle_score in zip(
            top_sequences, top_surrogate_scores, top_oracle_scores
        ):
            # Convert to string for sequence field
            seq_str = decode_sequence(seq_encoded, alphabet=list(self.token_to_idx.keys()))
            
            record = TrajectoryRecord(
                sequence=seq_str,
                oracle_score=float(oracle_score),
                surrogate_score=float(surrogate_score),
                method="rl",
                iteration=iteration,
                run_id=self.run_id,
                seed=self.seed,
                min_hamming_distance=0,  # Will be filled in
                dataset=self.config.dataset,
                transcription_factor=self.rl_config.transcription_factor,
            )
            records.append(record)
        
        # Compute Hamming distances
        if records:
            sequences_str = [r.sequence for r in records]
            distances = self.compute_hamming_distance(sequences_str)
            for record, distance in zip(records, distances):
                record.min_hamming_distance = distance
        
        # Retrain LSTM on selected + replay
        self._retrain_lstm_with_replay(top_sequences, top_surrogate_scores)
        
        print(f"iter={iteration} seed={self.seed} max_score={self.max_score:.4f} "
                f"rng_hash={hash(torch.get_rng_state().numpy().tobytes())}")

        return records
    
    def _retrain_lstm_with_replay(self, selected_sequences, selected_scores):
        """Retrain LSTM with selected sequences and replay from training data."""
        x_new = torch.LongTensor(selected_sequences)
        y_new = torch.FloatTensor(selected_scores).view(-1, 1)
        
        # Compute replay amount
        n_replay = int(len(x_new) * self.rl_config.replay_fraction / (1 - self.rl_config.replay_fraction))
        n_replay = min(n_replay, len(self.X_train))
        
        # Sample from training data
        replay_x = torch.LongTensor(self.X_train)
        replay_y = torch.FloatTensor(self.y_train).view(-1, 1)
        
        replay_idx = torch.randperm(len(replay_x))[:n_replay]
        
        # Combine
        x_combined = torch.cat([x_new, replay_x[replay_idx]])
        y_combined = torch.cat([y_new, replay_y[replay_idx]])
        
        # Create dataloader and train
        dataloader = DataLoader(
            TensorDataset(x_combined, y_combined),
            batch_size=64,
            shuffle=True
        )
        
        self._train_goal_directed_model(
            dataloader,
            epochs=20,
            lr=self.rl_config.lr,
            entropy_weight=self.rl_config.entropy_weight
        )
