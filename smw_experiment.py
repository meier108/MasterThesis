"""Single Mutant Walker experiment implementation."""

from typing import List
import numpy as np

from base_experiment import BaseExperiment
from trajectory import TrajectoryRecord
from experiment_config import ExperimentConfig
from models import oracle, single_mutant_walker, mlp
from assets.data_ops import load_data, build_tfbind8_dataframe, encode_sequence, one_hot_encode_sequence


class SMWExperiment(BaseExperiment):
    """Single Mutant Walker optimization experiment."""
    
    def __init__(self, config: ExperimentConfig, run_id: int):
        super().__init__(config, run_id)
        
        # SMW-specific state
        self.walker = None
        self.current_sequence = None
        self.current_surrogate = None
        self.current_oracle = None
        self.seen_sequences = None
        self.smw_config = config.smw_config
        
    def setup(self):
        """Initialize oracle, surrogate, walker, and training data."""
        np.random.seed(self.seed)
        
        # Load data based on dataset type
        if self.config.dataset == "tfbind8":
            self._setup_tfbind8()
        elif self.config.dataset == "gb1":
            self._setup_gb1()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        # Initialize surrogate model
        model = mlp.MLPModel()
        self.surrogate = self._fit_surrogate(model)
        
        # Initialize walker
        seq_length = len(self.train_df["sequence"].iloc[0])
        self.walker = single_mutant_walker.SingleMutantWalker(
            list(self.token_to_idx.keys()), 
            seq_length
        )
        
        # Prepare training data for Hamming distance computation
        self.X_train_encoded = np.array([
            encode_sequence(seq, self.token_to_idx) 
            for seq in self.train_df["sequence"]
        ])
        
        # Initialize current sequence
        self._initialize_seed_sequence()
        
        # Track seen sequences to avoid revisits
        self.seen_sequences = set(self.train_df["sequence"])
    
    def _setup_tfbind8(self):
        """Setup TFBind8 dataset."""
        alphabet = ["A", "C", "G", "T"]
        self.token_to_idx = {token: idx for idx, token in enumerate(alphabet)}
        
        x, y = load_data(name="tfbind8")
        df = build_tfbind8_dataframe(x, y, alphabet)
        self.train_df = df[df["split"] == "train"].copy()
        
        self.oracle = oracle.Oracle_TFBind8(df)
    
    def _setup_gb1(self):
        """Setup GB1 dataset."""
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.token_to_idx = {token: idx for idx, token in enumerate(alphabet)}
        
        df = load_data(name="gb1")
        self.train_df = df[df["split"] == "train"].copy()
        
        # Get sequence length
        one_hot_sequence = one_hot_encode_sequence(
            encode_sequence(self.train_df['sequence'].iloc[0], self.token_to_idx),
            num_tokens=len(self.token_to_idx)
        )
        L = one_hot_sequence.shape[0]
        
        self.oracle = oracle.Oracle_GB1(L, token_to_idx=self.token_to_idx, seed=self.seed)
        # TODO: maybe load the pretrained oracle here -> more realistic scores than the random initialized one
        

        # Score training sequences with oracle
        self.train_df["binding_scores"] = self.oracle.score_batch(self.train_df["sequence"])
    
    def _fit_surrogate(self, model) -> object:
        """Train surrogate model on training data."""
        X_train = np.stack([
            encode_sequence(sequence, self.token_to_idx) 
            for sequence in self.train_df["sequence"]
        ])
        X_train_one_hot = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx)) 
            for seq in X_train
        ])
        y_train = self.train_df["binding_scores"].to_numpy(dtype=np.float32)
        
        model.fit(X_train_one_hot, y_train)
        return model
    
    def _predict_surrogate(self, sequence: str) -> float:
        """Predict surrogate score for a sequence."""
        encoded = encode_sequence(sequence, self.token_to_idx).reshape(1, -1)
        one_hot_encoded = one_hot_encode_sequence(
            encoded.flatten(), 
            num_tokens=len(self.token_to_idx)
        ).reshape(1, -1)
        prediction = self.surrogate.predict(one_hot_encoded)
        return float(np.asarray(prediction).reshape(-1)[0])
    
    def _initialize_seed_sequence(self):
        """Select initial seed sequence."""
        if self.smw_config.change_seed_sequence:
            # Choose from top quantile of training data
            threshold = self.train_df["binding_scores"].quantile(self.smw_config.seed_quantile)
            seed_sequence = self.train_df[
                self.train_df["binding_scores"] >= threshold
            ]["sequence"].sample(n=1, random_state=self.seed).iloc[0]
        else:
            seed_sequence = self.train_df["sequence"].iloc[0]
        
        self.current_sequence = seed_sequence
        self.current_surrogate = self._predict_surrogate(seed_sequence)
        self.current_oracle = self.oracle.evaluate(seed_sequence)
    
    def run_single_iteration(self, iteration: int) -> List[TrajectoryRecord]:
        """
        Run a single round of mutation and selection.
        
        Returns records for all improvements found in this iteration.
        """
        records = []
        
        # Include seed in iteration 0
        if iteration == 0:
            record = TrajectoryRecord(
                sequence=self.current_sequence,
                oracle_score=self.current_oracle,
                surrogate_score=self.current_surrogate,
                method="smw",
                iteration=0,
                run_id=self.run_id,
                seed=self.seed,
                min_hamming_distance=1,
                dataset=self.config.dataset,
                transcription_factor=self.smw_config.transcription_factor,
            )
            records.append(record)
        
        # Generate mutants
        mutants = [
            self.walker.mutate_sequence(self.current_sequence) 
            for _ in range(self.smw_config.mutants_per_round)
        ]
        
        # Evaluate each mutant
        for sequence in mutants:
            # Skip already seen sequences
            if sequence in self.seen_sequences:
                continue
            self.seen_sequences.add(sequence)
            
            # Skip sequences not in target split (TFBind8 only)
            if self.smw_config.target_split is not None:
                split_name = self.oracle.get_split(sequence)
                if split_name != self.smw_config.target_split:
                    continue
            
            # Evaluate oracle
            oracle_score = self.oracle.evaluate(sequence)
            if oracle_score is None:
                continue
            
            # Evaluate surrogate
            surrogate_score = self._predict_surrogate(sequence)
            
            # Keep if improvement
            if surrogate_score > self.current_surrogate:
                self.current_sequence = sequence
                self.current_surrogate = surrogate_score
                self.current_oracle = oracle_score
                
                record = TrajectoryRecord(
                    sequence=sequence,
                    oracle_score=oracle_score,
                    surrogate_score=surrogate_score,
                    method="smw",
                    iteration=iteration + 1,  # +1 because iteration 0 is seed
                    run_id=self.run_id,
                    seed=self.seed,
                    min_hamming_distance=1,
                    dataset=self.config.dataset,
                    transcription_factor=self.smw_config.transcription_factor,
                )
                records.append(record)
        
        # Compute Hamming distances for all records
        if records:
            sequences = [r.sequence for r in records]
            distances = self.compute_hamming_distance(sequences)
            for record, distance in zip(records, distances):
                record.min_hamming_distance = distance
        
        return records
