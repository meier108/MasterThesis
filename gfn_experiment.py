import numpy as np
import torch
import torch.nn as nn

from base_experiment import BaseExperiment
from trajectory import TrajectoryRecord
from experiment_config import ExperimentConfig
from models import gflownet, oracle, mlp
from assets.data_ops import load_data, build_tfbind8_dataframe, encode_sequence, one_hot_encode_sequence, decode_sequence


class GFlowNetExperiment(BaseExperiment):
    '''
    GFlowNet offline MBO: GFlowNet generator guided by a shared MLP surrogate.

    Consistent with SMW and RL: no oracle feedback during optimization.
    The surrogate is the same MLPModel used by SMW/RL, trained on the initial
    training split (ground-truth labels). Each iteration it is re-fitted on the
    growing dataset where new candidate scores come from the current surrogate
    (not the oracle).

    Outer loop (single iteration):
    1. Re-fit MLP surrogate on current dataset
    2. Train GFlowNet with reward R(x) = surrogate(x)^β using TB-loss,
       mixing online + offline trajectories with γ=0.5
    3. Generate t*K candidates, pick top K by surrogate score
    4. Score candidates with surrogate, add to dataset
    5. Evaluate with oracle for logging only
    6. Log TrajectoryRecords
    '''

    def __init__(self, config: ExperimentConfig, run_id: int):
        super().__init__(config, run_id)
        self.gfn_config = config.gfn_config

        # GFlowNet model
        self.gfn = None

        # Shared MLP surrogate (same as SMW/RL)
        self.surrogate = None

        # Growing dataset: ground-truth labels for training split,
        # surrogate scores for all subsequently added candidates
        self.data_X = None
        self.data_y = None

    def setup(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.config.dataset == "tfbind8":
            self._setup_tfbind8()
        elif self.config.dataset == "gb1":
            self._setup_gb1()
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")

        seq_length = len(self.train_df["sequence"].iloc[0])
        vocab_size = len(self.token_to_idx)

        # Fit MLP surrogate on ground-truth training labels
        self.surrogate = mlp.MLPModel()
        X_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=vocab_size)
            for seq in self.data_X
        ])
        self.surrogate.fit(X_oh, self.data_y)

        self.gfn = gflownet.GFlowNetMLP(
            vocab_size = vocab_size,
            seq_length = seq_length,
            hidden_dim = self.gfn_config.hidden_dim
        )

    def _setup_tfbind8(self):
        alphabet = ["A", "C", "G", "T"]
        self.token_to_idx = {token: idx for idx, token in enumerate(alphabet)}
        x, y = load_data(name="tfbind8")
        df = build_tfbind8_dataframe(x, y, alphabet)
        self.train_df = df[df["split"] == "train"].copy()
        self.oracle = oracle.Oracle_TFBind8(df)

        self.X_train_encoded = np.array([
            encode_sequence(seq, self.token_to_idx) 
            for seq in self.train_df["sequence"]
        ])
        self.data_X = self.X_train_encoded.copy()
        self.data_y = self.train_df["binding_scores"].values.astype(np.float32)

    def _setup_gb1(self):
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        self.token_to_idx = {t: i for i, t in enumerate(alphabet)}
 
        df = load_data(name="gb1")
        self.train_df = df[df["split"] == "train"].copy()

        self.oracle = oracle.load_GB1_oracle()

        self.X_train_encoded = np.array([
            encode_sequence(seq, self.token_to_idx) 
            for seq in self.train_df["sequence"]
        ])
        self.data_X = self.X_train_encoded.copy()
        self.data_y = np.array(self.oracle.score_batch(self.train_df["sequence"]), dtype=np.float32)

    ### Surrogate training

    def _fit_surrogate(self):
        X_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx))
            for seq in self.data_X
        ])
        self.surrogate.fit(X_oh, self.data_y)
    
    ### GFlowNet training

    def _trajectory_balace_loss(self, sequences: torch.LongTensor, rewards: torch.Tensor) -> torch.Tensor:
        '''
        Trajectory Balance objective (Malkin et al. 2022, Eq. 11):
 
            L_TB(τ; θ) = ( log Z_θ + log P_F(τ; θ) - log R(x) )²
 
        Args:
            sequences: (batch, seq_len) token indices
            rewards  : (batch,) positive reward values R(x)
        Returns:
            scalar loss
        '''
        log_pf = self.gfn.get_log_pf(sequences)
        log_r = torch.log(rewards.clamp(min=1e-8))
        log_z = self.gfn.log_z

        loss = (log_z + log_pf - log_r) ** 2
        return loss.mean()
    
    def _get_reward(self, sequences_encoded: np.ndarray) -> np.ndarray:
        '''
        Compute reward R(x) = surrogate(x)^β for an array of encoded sequences.
        Scores are shifted to be strictly positive before exponentiation.
        '''
        X_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx))
            for seq in sequences_encoded
        ])
        scores = self.surrogate.predict(X_oh).astype(np.float32)
        scores_shifted = scores - scores.min() + 1e-3
        rewards = np.power(scores_shifted, self.gfn_config.beta).astype(np.float32)
        return rewards
    
    def _offline_trajectories(self, n : int) -> tuple:
        '''
        Sample n trajectories from the current dataset (offline data).
        Output:
            (sequences_tensor, rewards_tensor)
        '''
        idx = np.random.choice(len(self.data_X), size=n, replace=(n > len(self.data_X)))
        seqs = self.data_X[idx]
        rewards = self.data_y[idx]
        return torch.LongTensor(seqs), torch.FloatTensor(rewards)
    
    def _train_gflownet(self):
        '''
        Train GFlowNet for T steps using TB-loss with gamma-offline mixing.

        For each training step:
        1. Sample a batch of online trajectories from the current GFlowNet policy.
        2. Sample a batch of offline trajectories from the dataset.
        3. Compute rewards for both sets of trajectories using the surrogate.
        '''
        cfg = self.gfn_config
        m = cfg.minibatch_size
        n_offline = int(m*cfg.gamma)
        n_online = m - n_offline

        optimizer = torch.optim.Adam(
            [
                {"params": self.gfn.net.parameters(), "lr": cfg.gfn_lr},
                {"params": self.gfn.log_z, "lr": cfg.log_z_lr}
            ], betas = (0.9, 0.999)
        )

        self.gfn.train()
        for step in range(cfg.gfn_train_steps):
            optimizer.zero_grad()
            online_seqs = self.gfn.sample_sequence(n_online, temperature=1.0, delta = cfg.delta)
            online_rewards = self._get_reward(online_seqs.numpy())
            online_rewards_t = torch.FloatTensor(online_rewards)

            offline_seqs, offline_rewards_t = self._offline_trajectories(n_offline)

            all_seqs = torch.cat([online_seqs, offline_seqs], dim=0)
            all_rewards = torch.cat([online_rewards_t, offline_rewards_t], dim=0)

            loss = self._trajectory_balace_loss(all_seqs, all_rewards)
            loss.backward()
            optimizer.step()

            if (step +1) % 500 == 0:
                print(f"Step {step+1}/{cfg.gfn_train_steps}, TB Loss: {loss.item():.4f}, LogZ: {self.gfn.log_z.item():.4f}")

    ### Candidate generation

    def _generate_candidates(self) -> tuple:
        '''
        Sample (batch_size / top_k_ratio) candidates, return top batch_size by surrogate score.
        Returns (candidates_encoded, surrogate_scores) for the selected top-k.
        '''
        cfg = self.gfn_config
        n_samples = int(cfg.batch_size / cfg.top_k_ratio)

        generated = self.gfn.sample_sequence(n_samples, temperature=1.0, delta=cfg.delta).numpy()

        X_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx))
            for seq in generated
        ])
        surrogate_scores = self.surrogate.predict(X_oh).astype(np.float32)

        top_k_idx = np.argsort(surrogate_scores)[::-1][:cfg.batch_size]
        return generated[top_k_idx], surrogate_scores[top_k_idx]
    
    def run_single_iteration(self, iteration: int):
        records = []
        cfg = self.gfn_config
        alphabet = list(self.token_to_idx.keys())

        print(f"\nGFlowNet Iteration {iteration + 1}/{cfg.num_iterations}")
        print(f"  Dataset size: {len(self.data_X)}")

        # Step 1: Re-fit MLP surrogate on current dataset
        print("  Fitting surrogate...")
        self._fit_surrogate()

        # Step 2: Train GFlowNet
        print(f"  Training GFlowNet ({cfg.gfn_train_steps} steps)...")
        self._train_gflownet()

        # Step 3: Generate candidates, scored by surrogate (no redundant forward pass)
        print(f"  Generating candidates (batch_size={cfg.batch_size})...")
        candidates, surrogate_scores = self._generate_candidates()

        candidate_seqs_str = [decode_sequence(seq, alphabet=alphabet) for seq in candidates]

        # Step 4: Oracle evaluation for logging only — not fed back into training
        oracle_scores = []
        for seq_str in candidate_seqs_str:
            score = self.oracle.evaluate(seq_str)
            oracle_scores.append(float(score) if score is not None else 0.0)
        oracle_scores = np.array(oracle_scores, dtype=np.float32)

        # Step 5: Accumulate candidates with surrogate scores (not oracle)
        self.data_X = np.concatenate([self.data_X, candidates], axis=0)
        self.data_y = np.concatenate([self.data_y, surrogate_scores], axis=0)

        # Step 6: Build TrajectoryRecords
        hamming_distances = self.compute_hamming_distance(candidate_seqs_str)

        for seq_str, surrogate_score, oracle_score, hamming_dist in zip(
            candidate_seqs_str, surrogate_scores, oracle_scores, hamming_distances
        ):
            record = TrajectoryRecord(
                sequence=seq_str,
                oracle_score=float(oracle_score),
                surrogate_score=float(surrogate_score),
                method="gfn",
                iteration=iteration,
                run_id=self.run_id,
                seed=self.seed,
                min_hamming_distance=int(hamming_dist),
                dataset=self.config.dataset,
            )
            records.append(record)

        print(
            f"  Oracle scores — max: {oracle_scores.max():.4f}, mean: {oracle_scores.mean():.4f}"
        )
        return records