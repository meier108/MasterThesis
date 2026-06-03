import os
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from data import gb1_data 
from models.oracle import load_GB1_oracle

from models.random_forest import RandomForestModel
from models.mlp import MLPModel
from experiment_loop import fit_surrogate_model, predict_surrogate
from assets.data_ops import one_hot_encode_sequence, encode_sequence
from tqdm.auto import tqdm


def calculate_spearmanr(y_true, y_pred):
    """Calculate Spearman's rank correlation coefficient."""
    return spearmanr(y_true, y_pred).correlation

def calculate_mse(y_true, y_pred):    
    """Calculate Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)

def calculate_bias(y_true, y_pred):
    """Calculate Bias."""
    return np.mean(y_pred - y_true)

def load_generated_sequences(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Generated sequences file not found: {file_path}. Please run the sequence generation script first or set generate_sequences=True.")
    return pd.read_csv(file_path)


def score_generated_sequences(df_generated: pd.DataFrame, oracle: 'Oracle_GB1') -> pd.DataFrame:
    df_generated['oracle_score'] = oracle.score_batch(df_generated['sequence'].tolist())
    return df_generated


def main():
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    token_to_idx = {token: idx for idx, token in enumerate(ALPHABET)}

    df  = gb1_data.load_gb1_dataframe()
    
    
    oracle = load_GB1_oracle()

    # Train Surrogate Models
    train_df = df[df["split"] == "train"].copy()
    one_hot_sequence = one_hot_encode_sequence(encode_sequence(train_df['sequence'].iloc[0], token_to_idx), num_tokens=len(token_to_idx))
    train_df["binding_scores"] = oracle.score_batch(train_df["sequence"])

    rf = RandomForestModel()
    mlp = MLPModel()

    
    rf = fit_surrogate_model(rf, train_df, token_to_idx)
    mlp = fit_surrogate_model(mlp, train_df, token_to_idx)

    df_generated = load_generated_sequences('data/gb1_generated_sequences.csv')
    df_generated = score_generated_sequences(df_generated, oracle)

    for seq in tqdm(df_generated['sequence']):
        score_rf = predict_surrogate(rf, seq, token_to_idx)
        score_mlp = predict_surrogate(mlp, seq, token_to_idx)
        df_generated.loc[df_generated['sequence'] == seq, 'rf_prediction'] = score_rf
        df_generated.loc[df_generated['sequence'] == seq, 'mlp_prediction'] = score_mlp

    df_generated.to_csv('results/gb1_generated_sequences_scored.csv', index=False)


        

