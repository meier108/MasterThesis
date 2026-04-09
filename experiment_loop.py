from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from data import create_split, tfbind8_data
from models import oracle, random_forest, single_mutant_walker


def decode_sequence(encoded: np.ndarray, alphabet: List[str]) -> str:
    return "".join(alphabet[int(token)] for token in encoded)


def encode_sequence(sequence: str, token_to_idx: Dict[str, int]) -> np.ndarray:
    return np.array([token_to_idx[token] for token in sequence], dtype=np.int32)


def load_tfbind8_data(
    transcription_factor: str = "SIX6_REF_R1",
    local_data_dir: str = "data/design_bench_data",
):
    return tfbind8_data.load_tfbind8(
        transcription_factor=transcription_factor,
        local_dir=local_data_dir,
    )


def build_dataset_dataframe(x: np.ndarray, y: np.ndarray, alphabet: List[str]) -> pd.DataFrame:
    y = np.asarray(y).reshape(-1).astype(np.float32)
    sequences = [decode_sequence(row, alphabet) for row in x]

    df = pd.DataFrame(
        {
            "sequence": sequences,
            "binding_scores": y,
            "split": ["None"] * len(sequences),
        }
    )
    df = create_split.create_split(df)
    df["sequence"] = df["sequence"].str.upper()
    return df


def fit_surrogate_model(model, train_df: pd.DataFrame, token_to_idx: Dict[str, int]):
    x_train = np.stack(
        [encode_sequence(sequence, token_to_idx) for sequence in train_df["sequence"]]
    )
    y_train = train_df["binding_scores"].to_numpy(dtype=np.float32)
    model.fit(x_train, y_train)
    return model


def predict_surrogate(model, sequence: str, token_to_idx: Dict[str, int]) -> float:
    encoded = encode_sequence(sequence, token_to_idx).reshape(1, -1)
    prediction = model.predict(encoded)
    return float(np.asarray(prediction).reshape(-1)[0])


def compute_split_metrics(model, df: pd.DataFrame, token_to_idx: Dict[str, int], split_name: str):
    split_df = df[df["split"] == split_name]
    if split_df.empty:
        return {
            "split": split_name,
            "n": 0,
            "spearman_rho": np.nan,
            "mse": np.nan,
            "bias": np.nan,
        }

    x_split = np.stack(
        [encode_sequence(sequence, token_to_idx) for sequence in split_df["sequence"]]
    )
    y_true = split_df["binding_scores"].to_numpy(dtype=np.float32)
    y_pred = np.asarray(model.predict(x_split)).reshape(-1)

    rho = spearmanr(y_true, y_pred).correlation
    mse = mean_squared_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    return {
        "split": split_name,
        "n": int(len(split_df)),
        "spearman_rho": float(rho) if rho is not None else np.nan,
        "mse": float(mse),
        "bias": bias,
    }


def run_mutation_trajectory(
    seed_sequence: str,
    sequence_oracle,
    surrogate_model,
    walker,
    token_to_idx: Dict[str, int],
    target_split: str,
    num_rounds: int,
    mutants_per_round: int,
):
    current_sequence = seed_sequence
    current_surrogate = predict_surrogate(surrogate_model, current_sequence, token_to_idx)
    current_oracle = sequence_oracle.evaluate(current_sequence)

    trajectory = []
    total_candidates = 0
    filtered_candidates = 0

    for round_idx in range(1, num_rounds + 1):
        mutants = [walker.mutate_sequence(current_sequence) for _ in range(mutants_per_round)]
        total_candidates += len(mutants)

        valid_candidates = []
        for sequence in mutants:
            split_name = sequence_oracle.get_split(sequence)
            if split_name != target_split:
                continue

            oracle_score = sequence_oracle.evaluate(sequence)
            if oracle_score is None:
                continue

            surrogate_score = predict_surrogate(surrogate_model, sequence, token_to_idx)
            valid_candidates.append(
                {
                    "sequence": sequence,
                    "split": split_name,
                    "surrogate_score": float(surrogate_score),
                    "oracle_score": float(oracle_score),
                }
            )

        filtered_candidates += len(valid_candidates)
        if not valid_candidates:
            continue

        best = max(valid_candidates, key=lambda row: row["surrogate_score"])
        if best["surrogate_score"] > current_surrogate:
            current_sequence = best["sequence"]
            current_surrogate = best["surrogate_score"]
            current_oracle = best["oracle_score"]
            trajectory.append(
                {
                    "round": round_idx,
                    "sequence": current_sequence,
                    "surrogate_score": current_surrogate,
                    "oracle_score": current_oracle,
                    "split": target_split,
                }
            )

    summary = {
        "seed_sequence": seed_sequence,
        "seed_oracle_score": float(sequence_oracle.evaluate(seed_sequence)),
        "final_sequence": current_sequence,
        "final_surrogate_score": float(current_surrogate),
        "final_oracle_score": float(current_oracle),
        "target_split": target_split,
        "num_rounds": num_rounds,
        "mutants_per_round": mutants_per_round,
        "total_candidates": total_candidates,
        "valid_candidates_in_target_split": filtered_candidates,
        "num_improvements": len(trajectory),
    }

    return {"trajectory": pd.DataFrame(trajectory), "summary": summary}


def run_tfbind8_experiment(
    model=None,
    seed: int = 42,
    transcription_factor: str = "SIX6_REF_R1",
    target_split: str = "test_c",
    num_rounds: int = 30,
    mutants_per_round: int = 32,
):
    np.random.seed(seed)

    alphabet = ["A", "C", "G", "T"]
    token_to_idx = {token: idx for idx, token in enumerate(alphabet)}

    x, y = load_tfbind8_data(transcription_factor=transcription_factor)
    df = build_dataset_dataframe(x, y, alphabet)
    train_df = df[df["split"] == "train"].copy()

    if model is None:
        model = random_forest.RandomForestModel(n_estimators=200, random_state=seed)

    model = fit_surrogate_model(model, train_df, token_to_idx)

    sequence_oracle = oracle.Oracle_TFBind8(df)
    seed_sequence = train_df["sequence"].iloc[0]
    walker = single_mutant_walker.SingleMutantWalker(alphabet, len(seed_sequence))

    metrics = {
        split: compute_split_metrics(model, df, token_to_idx, split)
        for split in ["test_a", "test_b", "test_c"]
    }

    trajectory_result = run_mutation_trajectory(
        seed_sequence=seed_sequence,
        sequence_oracle=sequence_oracle,
        surrogate_model=model,
        walker=walker,
        token_to_idx=token_to_idx,
        target_split=target_split,
        num_rounds=num_rounds,
        mutants_per_round=mutants_per_round,
    )

    return {
        "df": df,
        "train_df": train_df,
        "model": model,
        "metrics": metrics,
        "trajectory": trajectory_result,
    }
