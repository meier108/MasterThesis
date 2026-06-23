"""Configuration classes for experiments."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SMWConfig:
    """Configuration for Single Mutant Walker experiment."""
    
    transcription_factor: Optional[str] = "SIX6_REF_R1"  # None for GB1
    target_split: Optional[str] = None  # None for GB1, also removed target splits from TFBind8 for better comparison
    num_iterations: int = 20  # Changed from 100 to 20
    mutants_per_round: int = 10
    change_seed_sequence: bool = True
    seed_quantile: float = 0.5


@dataclass
class RLConfig:
    """Configuration for Reinforcement Learning experiment."""
    
    transcription_factor: Optional[str] = "SIX6_REF_R1"  # None for GB1
    num_iterations: int = 20  # Changed from 6 to 20
    batch_size: int = 128  # for LSTM generation
    min_hamming: int = 2
    replay_fraction: float = 0.2
    temp_start: float = 0.5
    temp_end: float = 1.0
    entropy_weight: float = 0.01
    lstm_epochs: int = 30
    lr: float = 1e-3

@dataclass
class GFlowNetConfig:

    transcription_factor: Optional[str] = "SIX6_REF_R1"  # None for GB1
    num_iterations: int = 15
    batch_size: int = 64

    # GFlowNet architecture + training
    hidden_dim: int = 2048
    gfn_lr: float = 1e-5
    log_z_lr: float = 1e-3
    gfn_train_steps: int = 5000
    minibatch_size: int = 32
    delta: float = 0.001
    beta: float = 3.0
    gamma: float = 0.5
    top_k_ratio: float = 0.8

@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    
    method: str  # "smw" or "rl"
    dataset: str  # "tfbind8" or "gb1"
    seed: int = 42
    num_runs: int = 10
    
    # Method-specific configs
    smw_config: Optional[SMWConfig] = None
    rl_config: Optional[RLConfig] = None
    gfn_config: Optional[GFlowNetConfig] = None
    
    def __post_init__(self):
        """Initialize method-specific configs if not provided."""
        if self.smw_config is None:
            self.smw_config = SMWConfig()
        if self.rl_config is None:
            self.rl_config = RLConfig()
        if self.gfn_config is None:
            self.gfn_config = GFlowNetConfig()

# Predefined experiment keywords
EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "smw_tfbind8": ExperimentConfig(
        method="smw",
        dataset="tfbind8",
        smw_config=SMWConfig(
            transcription_factor="SIX6_REF_R1",
            target_split=None,
            num_iterations=15,
            mutants_per_round=10,
        ),
    ),
    "smw_gb1": ExperimentConfig(
        method="smw",
        dataset="gb1",
        smw_config=SMWConfig(
            transcription_factor=None,
            target_split=None,
            num_iterations=15,
            mutants_per_round=10,
        ),
    ),
    "rl_tfbind8": ExperimentConfig(
        method="rl",
        dataset="tfbind8",
        rl_config=RLConfig(
            transcription_factor="SIX6_REF_R1",
            num_iterations=15,
            batch_size=265,
        ),
    ),
    "rl_gb1": ExperimentConfig(
        method="rl",
        dataset="gb1",
        rl_config=RLConfig(
            transcription_factor=None,
            num_iterations=15,
            batch_size=265, 
        ),
    ),
    "gfn_tfbind8": ExperimentConfig(
        method='gfn',
        dataset='tfbind8',
        gfn_config = GFlowNetConfig(
            transcription_factor='SIX6_REF_R1',
            num_iterations=15,
            batch_size=64,
        ),
    ),
    "gfn_gb1": ExperimentConfig(
        method='gfn', 
        dataset = 'gb1',
        gfn_config = GFlowNetConfig(
            transcription_factor=None,
            num_iterations=15,
            batch_size=256,
        ),
    ),
}


def get_config(keyword: str) -> ExperimentConfig:
    """Retrieve experiment config by keyword."""
    if keyword not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment keyword: {keyword}. "
            f"Available: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    return EXPERIMENT_CONFIGS[keyword]
