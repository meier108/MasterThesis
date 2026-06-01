"""Abstract base class for experiments."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd

from trajectory import TrajectoryRecord
from experiment_config import ExperimentConfig
from assets.compute_metrics import min_hamming_distance
from assets.data_ops import encode_sequence


class BaseExperiment(ABC):
    """Abstract base class for all optimization experiments."""
    
    def __init__(self, config: ExperimentConfig, run_id: int):
        """
        Initialize experiment.
        
        Args:
            config: ExperimentConfig instance
            run_id: Which run is this (1-10)
        """
        self.config = config
        self.run_id = run_id
        self.seed = config.seed + run_id  # Vary seed per run
        
        # Will be set by setup()
        self.oracle = None
        self.surrogate = None
        self.train_df = None
        self.token_to_idx = None
        self.X_train_encoded = None  # For Hamming distance computation
        
    def setup(self):
        """Setup experiment: initialize oracle, surrogate, training data."""
        raise NotImplementedError("Subclasses must implement setup()")
    
    def run_single_iteration(self, iteration: int) -> List[TrajectoryRecord]:
        """
        Run a single iteration and return trajectory records.
        
        Args:
            iteration: Iteration number (0-indexed)
            
        Returns:
            List of TrajectoryRecord for this iteration
        """
        raise NotImplementedError("Subclasses must implement run_single_iteration()")
    
    def compute_hamming_distance(self, sequences: List[str]) -> List[int]:
        """
        Compute minimum Hamming distance from sequences to training set.
        
        Args:
            sequences: List of sequence strings to evaluate
            
        Returns:
            List of minimum Hamming distances
        """
        if self.X_train_encoded is None:
            raise ValueError("X_train_encoded not set. Call setup() first.")
        
        # Convert sequences to encoded format
        X_test = np.array([
            encode_sequence(seq, self.token_to_idx) 
            for seq in sequences
        ])
        
        # Compute minimum Hamming distances
        distances = min_hamming_distance(X_test, self.X_train_encoded)
        return distances.tolist()
    
    def run(self) -> List[TrajectoryRecord]:
        """
        Run complete experiment for this run.
        
        Returns:
            List of TrajectoryRecord for all iterations
        """
        self.setup()
        
        all_records = []

        iterations = (self.config.smw_config.num_iterations if self.config.method == 'smw' 
              else self.config.rl_config.num_iterations if self.config.method == 'rl' 
              else self.config.gfn_config.num_iterations)
        
        for iteration in range(iterations):
            records = self.run_single_iteration(iteration)
            all_records.extend(records)
        
        return all_records
