from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from models import oracle, random_forest, single_mutant_walker
from assets.data_ops import load_data, build_tfbind8_dataframe, decode_sequence, encode_sequence, one_hot_encode_sequence


def fit_surrogate_model(model, train_df: pd.DataFrame, token_to_idx: Dict[str, int]):
    x_train = np.stack(
        [encode_sequence(sequence, token_to_idx) for sequence in train_df["sequence"]]
    )
    x_train_one_hot = np.stack([one_hot_encode_sequence(seq, num_tokens=len(token_to_idx)) for seq in x_train])
    y_train = train_df["binding_scores"].to_numpy(dtype=np.float32)
    model.fit(x_train_one_hot, y_train)
    return model

def predict_surrogate(model, sequence: str, token_to_idx: Dict[str, int]) -> float:
    encoded = encode_sequence(sequence, token_to_idx).reshape(1, -1)
    one_hot_encoded = one_hot_encode_sequence(encoded.flatten(), num_tokens=len(token_to_idx)).reshape(1, -1)
    prediction = model.predict(one_hot_encoded)
    return float(np.asarray(prediction).reshape(-1)[0])

def run_mutation_trajectory(
    seed_sequence: str,
    sequence_oracle,
    surrogate_model,
    walker,
    token_to_idx: Dict[str, int],
    target_split: str,
    num_rounds: int,
    mutants_per_round: int,
    seen_sequences: set,
    return_stats: bool = True,
):
    current_sequence = seed_sequence
    current_surrogate = predict_surrogate(surrogate_model, current_sequence, token_to_idx)
    current_oracle = sequence_oracle.evaluate(current_sequence)

    trajectory = []
    stats = {
        "generated": 0,
        "skipped_seen": 0,
        "skipped_wrong_split": 0,
        "in_target_split": 0,
        "invalid_oracle": 0,
        "surrogate_improvements": 0,
    }

    # Append seed sequence. 
    trajectory.append(
                    {
                        "round": 0,
                        "sequence": current_sequence,
                        "surrogate_score": current_surrogate,
                        "oracle_score": current_oracle,
                        "split": target_split,
                    }
                )
    
    for round_idx in range(1, num_rounds + 1):
        mutants = [walker.mutate_sequence(current_sequence) for _ in range(mutants_per_round)]
        stats["generated"] += len(mutants)

        for sequence in mutants:
            # Skip already seen sequences
            if seen_sequences is not None and sequence in seen_sequences:
                stats["skipped_seen"] += 1
                continue
            if seen_sequences is not None:
                seen_sequences.add(sequence)

            if target_split is not None:
                # Skip sequences not in the target split
                split_name = sequence_oracle.get_split(sequence)
                if split_name != target_split:
                    stats["skipped_wrong_split"] += 1
                    continue
                stats["in_target_split"] += 1
            
            # Skip sequences with invalid oracle scores -> should not happen
            oracle_score = sequence_oracle.evaluate(sequence)
            if oracle_score is None:
                stats["invalid_oracle"] += 1
                continue

            surrogate_score = predict_surrogate(surrogate_model, sequence, token_to_idx)
        
            if surrogate_score > current_surrogate:
                current_sequence = sequence
                current_surrogate = float(surrogate_score)
                current_oracle = float(oracle_score)
                stats["surrogate_improvements"] += 1
                trajectory.append(
                    {
                        "round": round_idx,
                        "sequence": current_sequence,
                        "surrogate_score": current_surrogate,
                        "oracle_score": current_oracle,
                        "split": target_split,
                    }
                )

    result = {"trajectory": pd.DataFrame(trajectory)}
    if return_stats:
        result["stats"] = stats
    return result

def setup_experiment(seed: int, transcription_factor: str, model=None):
    '''Set up a experiment either with TFBind8 or GB1 dataset, depending on the transcription factor.
    If transcription_factor is None, it will set up the GB1 experiment.'''
    
    np.random.seed(seed)
    token_to_idx = None

    ## Transcription factor -> TFBind8 dataset
    if transcription_factor is not None:
        alphabet = ["A", "C", "G", "T"]
        token_to_idx = {token: idx for idx, token in enumerate(alphabet)}

        x, y = load_data(name="tfbind8")
        df = build_tfbind8_dataframe(x, y, alphabet)
        train_df = df[df["split"] == "train"].copy()
        # Print range of train sequences
        #print(f"Training sequences: {len(train_df)}")
        #print(f"Training binding score range: {train_df['binding_scores'].min()} to {train_df['binding_scores'].max()}")
        oracle_ = oracle.Oracle_TFBind8(df)
    
    ## No transcription factor -> GB1 dataset
    else:
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        token_to_idx = {token: idx for idx, token in enumerate(alphabet)}

        df = load_data(name="gb1")
        train_df = df[df["split"] == "train"].copy()

        one_hot_sequence = one_hot_encode_sequence(encode_sequence(train_df['sequence'].iloc[0], token_to_idx), num_tokens=len(token_to_idx))
        L = one_hot_sequence.shape[0]
        oracle_ = oracle.Oracle_GB1(L, token_to_idx=token_to_idx, seed=seed)

        # score train sequences with oracle for surrogate training
        # one hot encode train sequences for oracle
        train_df["binding_scores"] = oracle_.score_batch(train_df["sequence"])

    if model is None:
        model = random_forest.RandomForestModel(n_estimators=200, random_state=seed)

    surrogate = fit_surrogate_model(model, train_df, token_to_idx)

    walker = single_mutant_walker.SingleMutantWalker(alphabet, len(train_df["sequence"].iloc[0]))
    return train_df, oracle_, walker, surrogate, token_to_idx

def run_tfbind8_experiment(
    model=None,
    seed: int = 42,
    transcription_factor: str = "SIX6_REF_R1",
    target_split: str = "test_c",
    num_rounds: int = 100,
    mutants_per_round: int = 10,
    change_seed_sequence: bool = True,
    seed_quantile: float = 0.8,
    return_stats: bool = True,
):
    train_df, oracle, walker, surrogate, token_to_idx = setup_experiment(
        seed=seed,
        transcription_factor=transcription_factor,
        model=model
    )

    if change_seed_sequence is True:
        # Choose a seed only from known training data (lab-realistic assumption).
        top_percent_threshold = train_df["binding_scores"].quantile(seed_quantile)
        seed_sequence = train_df[train_df["binding_scores"] >= top_percent_threshold]["sequence"].sample(
            n=1, random_state=seed
        ).iloc[0]
        print(f"Selected seed sequence: {seed_sequence} with oracle score: {oracle.evaluate(seed_sequence)}")
    else:
        seed_sequence = train_df["sequence"].iloc[0]

    seen_sequences = set(train_df["sequence"])

    trajectory_result = run_mutation_trajectory(
        seed_sequence=seed_sequence,
        sequence_oracle=oracle,
        surrogate_model=surrogate,
        walker=walker,
        token_to_idx=token_to_idx,
        target_split=target_split,
        num_rounds=num_rounds,
        mutants_per_round=mutants_per_round,
        seen_sequences=seen_sequences,
        return_stats=return_stats,
    )
    
    return trajectory_result

def run_gb1_experiment(
        seed: int = 42,
        num_rounds: int = 100,
        mutants_per_round: int = 10,
        change_seed_sequence: bool = True,
        seed_quantile: float = 0.8,
        return_stats: bool = True,
        model=None
):
    train_df, oracle, walker, surrogate, token_to_idx = setup_experiment(
        seed=seed,
        transcription_factor=None,
        model = model
    )
    if change_seed_sequence is True:
        # Choose a seed only from known training data (lab-realistic assumption).
        top_percent_threshold = train_df["binding_scores"].quantile(seed_quantile)
        seed_sequence = train_df[train_df["binding_scores"] >= top_percent_threshold]["sequence"].sample(
            n=1, random_state=seed
        ).iloc[0]
        print(f"Selected seed sequence: {seed_sequence} with oracle score: {oracle.evaluate(seed_sequence)}")
    else:
        seed_sequence = train_df["sequence"].iloc[0]
    seen_sequences = set(train_df["sequence"])
    trajectory_result = run_mutation_trajectory(
        seed_sequence=seed_sequence,
        sequence_oracle=oracle,
        surrogate_model=surrogate,
        walker=walker,
        token_to_idx=token_to_idx,
        target_split=None,  # No splits in GB1
        num_rounds=num_rounds,
        mutants_per_round=mutants_per_round,
        seen_sequences=seen_sequences,
        return_stats=return_stats,
    )
    return trajectory_result


def run_multiple_experiments(model = None, seed=42, test_sets=None, runs=10, change_seed_sequence=True, num_rounds=100, mutants_per_round=10, seed_quantile=0.2, return_stats=False):
    '''Runs multiple experiments for different test sets from TFBind8 Dataset and rounds, collecting trajectories.'''
    all_trajectories = []

    train_df, oracle, walker, surrogate, token_to_idx = setup_experiment(
            seed=seed,
            transcription_factor="SIX6_REF_R1",
            model=model
            )
    
    for test_set in test_sets:
        print(f"\nRunning experiment for {test_set}...")

        for run in range(1, runs + 1):
            print(f"Run {run}/{runs} for {test_set}...")

            
            if change_seed_sequence is True:
            # choose a seed from the upper quantile of the known training split
                top_percent_threshold = train_df["binding_scores"].quantile(seed_quantile)
                seed_sequence = train_df[train_df["binding_scores"] >= top_percent_threshold]["sequence"].sample(
                    n=1, random_state=seed + run
                ).iloc[0]
                print(f"Selected seed sequence: {seed_sequence} with oracle score: {oracle.evaluate(seed_sequence)}")
            else:
                seed_sequence = train_df["sequence"].iloc[0]
            
            seen_sequences = set(train_df["sequence"])

            result = run_mutation_trajectory(
                seed_sequence=seed_sequence,
                sequence_oracle=oracle,
                surrogate_model=surrogate,
                walker=walker,
                token_to_idx=token_to_idx,
                target_split=test_set,
                num_rounds=num_rounds,
                mutants_per_round=mutants_per_round,
                seen_sequences=seen_sequences,
                return_stats=return_stats,
            )

            if return_stats and "stats" in result:
                s = result["stats"]
                print(
                    "Stats:",
                    f"generated={s['generated']}",
                    f"in_target={s['in_target_split']}",
                    f"improvements={s['surrogate_improvements']}",
                    f"wrong_split={s['skipped_wrong_split']}",
                    f"seen={s['skipped_seen']}",
                )

            trajectory_df = result["trajectory"].copy()
            if not trajectory_df.empty:
                trajectory_df["run"] = run
                trajectory_df["evaluated_split"] = test_set
                all_trajectories.append(trajectory_df)
            else:
                print(f"Warning: Trajectory for run {run} on {test_set} is empty and will be skipped in the trajectories DataFrame.")
            
    return all_trajectories
