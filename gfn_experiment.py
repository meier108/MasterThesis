import numpy as np
import torch
import torch.nn as nn

from base_experiment import BaseExperiment
from trajectory import TrajectoryRecord
from experiment_config import ExperimentConfig
from models import gflownet, oracle
from assets.data_ops import load_data, build_tfbind8_dataframe, encode_sequence, one_hot_encode_sequence, decode_sequence


class GFlowNetExperiment(BaseExperiment):
    '''
    GFlowNet: Active Learning with GFlowNet generator + proxy ensemble.
    
    Source: Algorithm 1 + 2 from Jain et al., 2022.
    
    Outer loop (single iteration):
    1. Fit proxy ensemble on current data
    2. Train GFlowNet with reward R(x) = proxy_ucb(x)^β using TB-loss
        mixing online + offline (y) trajectories with γ=0.5
    3. Generate t*K candidates, pick top K by proxy UCB
    4. Evaluate candidates with oracle, add to dataset
    5. Log Trajectory Records

    '''

    def __init__(self, config: ExperimentConfig, run_id: int):
        super().__init__(config, run_id)
        self.gfn_config = config.gfn_config

        # Initialize GFlowNet components
        self.proxy_ensemble = None
        self.gfn = None
        self.gfn_optimizer = None

        # Growing dataset of (sequence, oracle_score) pairs
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
        input_dim = seq_length * vocab_size

        self.proxy = gflownet.ProxyEnsemble(
            input_dim = input_dim,
            n_members = self.gfn_config.n_proxy_members,
            hidden_dim = self.gfn_config.hidden_dim,
            kappa = self.gfn_config.kappa
        )
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

    ### Proxy training

    def _fit_proxy(self):
        X_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx)) 
            for seq in self.data_X
        ])
        self.proxy.fit(
            x = X_oh,
            y = self.data_y,
            epochs = self.gfn_config.proxy_epochs,
            lr = self.gfn_config.proxy_lr,
        )
    
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
        Compute reward R(x) = proxy_ucb(x)^β for an array of encoded sequences.
        Reward clipping is applied to get positive rewards.
        '''
        X_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx))
            for seq in sequences_encoded
        ])
        ucb_scores = self.proxy.ucb_numpy(X_oh)
        ucb_shifted = ucb_scores - ucb_scores.min() + 1e-3
        rewards = np.power(ucb_shifted, self.gfn_config.beta).astype(np.float32)
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
        3. Compute rewards for both sets of trajectories using the proxy UCB.
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
    def _generate_candidates(self) -> np.ndarray:
        '''
        Sample t*K candidates, return top K by proxy UCB.
        
        '''
        cfg = self.gfn_config
        n_samples = int(cfg.top_k_ratio * cfg.batch_size)

        generated = self.gfn.sample_sequence(n_samples, temperature=1.0, delta = cfg.delta).numpy()

        x_oh = np.stack([
            one_hot_encode_sequence(seq, num_tokens=len(self.token_to_idx)).flatten()
            for seq in generated
        ])

        ucb_scores = self.proxy.ucb_numpy(x_oh)
        top_k_idx = np.argsort(ucb_scores)[::-1][:cfg.batch_size]
        return generated[top_k_idx], ucb_scores[top_k_idx]
    
    def run_single_iteration(self, iteration: int):
        records = []
        cfg = self.gfn_config

        print(f"\nGFlowNet-AL Iteration {iteration + 1}/{cfg.num_iterations}")
        print(f"  Dataset size: {len(self.data_X)}")
 
        # Step 1: Fit proxy ensemble
        print("  Fitting proxy ensemble...")
        self._fit_proxy()
 
        # Step 2: Train GFlowNet
        print(f"  Training GFlowNet ({cfg.gfn_train_steps} steps)...")
        self._train_gflownet()
 
        # Step 3: Generate candidates
        print(f"  Generating candidates (batch_size={cfg.batch_size})...")
        candidates, proxy_ucb_scores = self._generate_candidates()

        # Step 4: Evaluate candidates with oracle
        alphabet = list(self.token_to_idx.keys())
        oracle_scores = []
        for seq_encoded in candidates:
            seq_str = decode_sequence(seq_encoded, alphabet=alphabet)
            score = self.oracle.evaluate(seq_str)
            oracle_scores.append(float(score) if score is not None else 0.0)
        oracle_scores = np.array(oracle_scores, dtype=np.float32)
 
        # Step 5: Compute proxy mean scores (μ, not UCB) for logging
        X_oh = np.stack(
            [one_hot_encode_sequence(s, num_tokens=len(self.token_to_idx)).flatten()
             for s in candidates]
        )
        proxy_mean, _ = self.proxy.predict(X_oh)
 
        # Step 6: Update dataset with new candidates
        self.data_X = np.concatenate([self.data_X, candidates], axis=0)
        self.data_y = np.concatenate([self.data_y, oracle_scores], axis=0)
 
        # Step 7: Build TrajectoryRecords
        candidate_seqs_str = [
            decode_sequence(seq, alphabet=alphabet) for seq in candidates
        ]
 
        # Compute Hamming distances to original training set
        hamming_distances = self.compute_hamming_distance(candidate_seqs_str)
 
        for seq_str, proxy_score, oracle_score, hamming_dist in zip(
            candidate_seqs_str, proxy_mean, oracle_scores, hamming_distances
        ):
            record = TrajectoryRecord(
                sequence=seq_str,
                oracle_score=float(oracle_score),
                surrogate_score=float(proxy_score),
                method="gfn",
                iteration=iteration,
                run_id=self.run_id,
                seed=self.seed,
                min_hamming_distance=int(hamming_dist),
                dataset=self.config.dataset,
            )
            records.append(record)
 
        print(
            f"  Oracle scores — max: {oracle_scores.max():.4f}, "
            f"mean: {oracle_scores.mean():.4f}"
        )
        return records