"""Unified trajectory record format for all experiments."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryRecord:
    """A single record in an experiment trajectory."""
    
    # Core scores
    sequence: str
    oracle_score: float
    surrogate_score: float
    
    # Metadata
    method: str  # "smw", "rl", etc.
    iteration: int  # 0-indexed iteration/round
    run_id: int  # which of the N runs (1-10)
    seed: int
    
    # Data about the sequence
    min_hamming_distance: int  # to training set
    
    # Dataset info
    dataset: str  # "tfbind8" or "gb1"
    transcription_factor: Optional[str] = None  # e.g., "SIX6_REF_R1" for tfbind8
    
    def to_dict(self):
        """Convert to dictionary for DataFrame conversion."""
        return {
            'method': self.method,
            'run_id': self.run_id,
            'seed': self.seed,
            'iteration': self.iteration,
            'sequence': self.sequence,
            'oracle_score': self.oracle_score,
            'surrogate_score': self.surrogate_score,
            'min_hamming_distance': self.min_hamming_distance,
            'dataset': self.dataset,
        }
